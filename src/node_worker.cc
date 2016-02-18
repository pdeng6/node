#include "node.h"
#include "node_worker.h"
#include "uv.h"
#include "libplatform/libplatform.h"
#include "env.h"
#include "env-inl.h"

#include "v8.h"

#include <string.h>
#include <unistd.h>

#define LOG(...)                 \
  do {                           \
    if (0) fprintf(__VA_ARGS__); \
  } while (0)

namespace node {
namespace Worker {

using v8::Array;
using v8::ArrayBufferCreationMode;
using v8::ArrayBufferView;
using v8::Context;
using v8::EscapableHandleScope;
using v8::External;
using v8::Float32Array;
using v8::Float64Array;
using v8::Function;
using v8::FunctionTemplate;
using v8::Global;
using v8::Handle;
using v8::HandleScope;
using v8::Int8Array;
using v8::Int16Array;
using v8::Int32Array;
using v8::Message;
using v8::NewStringType;
using v8::Number;
using v8::ObjectTemplate;
using v8::Script;
using v8::Signature;
using v8::TryCatch;
using v8::Uint16Array;
using v8::Uint32Array;
using v8::Uint8Array;
using v8::Uint8ClampedArray;

const static int kMaxWorkers = 50;
const static int kMaxInt = 0x7FFFFFFF;

static uv_mutex_t workers_mutex_;
static uv_mutex_t context_mutex_;
static bool allow_new_workers_ = true;
static std::vector<Worker*> workers_;
static std::vector<SharedArrayBuffer::Contents> externalized_shared_contents_;
static Global<Context> utility_context_;

typedef std::vector<Local<Object>> ObjectList;
static bool SerializeValue(Isolate* isolate, Local<Value> value,
                           const ObjectList& to_transfer,
                           ObjectList* seen_objects,
                           SerializationData* out_data);
static MaybeLocal<Value> DeserializeValue(Isolate* isolate,
                                          const SerializationData& data,
                                          int* offset);
static char* ReadChars(const char* name, int* size_out);
static const char* ToCString(const v8::String::Utf8Value& value);
static void WorkerNew(const v8::FunctionCallbackInfo<v8::Value>& args);
static void WorkerPostMessage(const v8::FunctionCallbackInfo<v8::Value>& args);
static void WorkerTerminate(const v8::FunctionCallbackInfo<v8::Value>& args);
static Handle<ObjectTemplate> CreateGlobalTemplate(Isolate* isolate);
static Local<Context> CreateEvaluationContext(Isolate* isolate);

// FIXME (Pan), It is ugly to init workers_mutex_ here, will be improved once
// uv_mutex_t wrapped in a Class and can be initialized in its ctor.
class WorkerMutexInitializer {
 public:
  WorkerMutexInitializer() {
    uv_mutex_init(&workers_mutex_);
    uv_mutex_init(&context_mutex_);
  }
};

WorkerMutexInitializer initializer;

class PerIsolateData {
 public:
  explicit PerIsolateData(Isolate* isolate) : isolate_(isolate) {
    HandleScope scope(isolate);
    isolate->SetData(0, this);
  }

  ~PerIsolateData() {
    isolate_->SetData(0, NULL);  // Not really needed, just to be sure...
  }

  inline static PerIsolateData* Get(Isolate* isolate) {
    return reinterpret_cast<PerIsolateData*>(isolate->GetData(0));
  }

 private:
  friend class Worker;
  Isolate* isolate_;
  Worker* worker_;
};


inline static int32_t NoBarrier_Load(volatile const int32_t* ptr) {
  return *ptr;
}

inline static void NoBarrier_Store(volatile int32_t* ptr, int32_t value) {
  *ptr = value;
}

static Local<Value> Throw(Isolate* isolate, const char* message) {
  return isolate->ThrowException(
      String::NewFromUtf8(isolate, message, NewStringType::kNormal)
          .ToLocalChecked());
}

Local<Context> CreateEvaluationContext(Isolate* isolate) {
  // This needs to be a critical section since this is not thread-safe
  uv_mutex_lock(&context_mutex_);

  // Initialize the global objects
  Local<ObjectTemplate> global_template = CreateGlobalTemplate(isolate);

  EscapableHandleScope handle_scope(isolate);
  Local<Context> context = Context::New(isolate, NULL, global_template);
  CHECK(!context.IsEmpty());

  uv_mutex_unlock(&context_mutex_);
  return handle_scope.Escape(context);
}

bool FindInObjectList(Local<Object> object, const ObjectList& list) {
  for (size_t i = 0; i < list.size(); ++i) {
    if (list[i]->StrictEquals(object)) {
      return true;
    }
  }
  return false;
}

const char* ToCString(const String::Utf8Value& value) {
  return *value ? *value : "<string conversion failed>";
}

Worker* GetWorkerFromInternalField(Isolate* isolate, Local<Object> object) {
  if (object->InternalFieldCount() != 1) {
    Throw(isolate, "this is not a Worker");
    return NULL;
  }

  Worker* worker =
      static_cast<Worker*>(object->GetAlignedPointerFromInternalField(0));
  if (worker == NULL) {
    Throw(isolate, "Worker is defunct because main thread is terminating");
    return NULL;
  }

  return worker;
}


void WorkerNew(const v8::FunctionCallbackInfo<v8::Value>& args) {
  Isolate* isolate = args.GetIsolate();
  HandleScope handle_scope(isolate);
  if (args.Length() < 1 || !args[0]->IsString()) {
    Throw(args.GetIsolate(), "1st argument must be string");
    return;
  }

  if (!args.IsConstructCall()) {
    Throw(args.GetIsolate(), "Worker must be constructed with new");
    return;
  }

  {
    uv_mutex_lock(&workers_mutex_);
    if (workers_.size() >= kMaxWorkers) {
      Throw(args.GetIsolate(), "Too many workers, I won't let you create more");
      uv_mutex_unlock(&workers_mutex_);
      return;
    }

    // Initialize the internal field to NULL; if we return early without
    // creating a new Worker (because the main thread is terminating) we can
    // early-out from the instance calls.
    args.Holder()->SetAlignedPointerInInternalField(0, NULL);

    if (!allow_new_workers_) {
      uv_mutex_unlock(&workers_mutex_);
      return;
    }

    Worker* worker = new Worker(args.This());
    args.Holder()->SetAlignedPointerInInternalField(0, worker);
    workers_.push_back(worker);

    if (worker->LoadWorkerScript(args)) worker->StartExecuteInThread();
    uv_mutex_unlock(&workers_mutex_);
  }
}

// Post message to worker
void WorkerPostMessage(const v8::FunctionCallbackInfo<v8::Value>& args) {
  Isolate* isolate = args.GetIsolate();
  HandleScope handle_scope(isolate);
  Local<Context> context = isolate->GetCurrentContext();

  if (args.Length() < 1) {
    Throw(isolate, "Invalid argument");
    return;
  }
  Worker* worker = GetWorkerFromInternalField(isolate, args.Holder());

  if (!worker || !worker->IsRunning()) {
    Throw(isolate, "Worker is not running anymore!");
    return;
  }

  Local<Value> message = args[0];
  ObjectList to_transfer;
  if (args.Length() >= 2) {
    if (!args[1]->IsArray()) {
      Throw(isolate, "Transfer list must be an Array");
      return;
    }

    Local<Array> transfer = Local<Array>::Cast(args[1]);
    uint32_t length = transfer->Length();
    for (uint32_t i = 0; i < length; ++i) {
      Local<Value> element;
      if (transfer->Get(context, i).ToLocal(&element)) {
        if (!element->IsArrayBuffer() && !element->IsSharedArrayBuffer()) {
          Throw(isolate,
                "Transfer array elements must be an ArrayBuffer or "
                "SharedArrayBuffer.");
          break;
        }

        to_transfer.push_back(Local<Object>::Cast(element));
      }
    }
  }

  ObjectList seen_objects;
  SerializationData* data = new SerializationData;
  if (SerializeValue(isolate, message, to_transfer, &seen_objects, data)) {
    worker->PostMessage(data);
  } else {
    delete data;
  }
}


void WorkerTerminate(const v8::FunctionCallbackInfo<v8::Value>& args) {
  Isolate* isolate = args.GetIsolate();
  HandleScope handle_scope(isolate);
  Worker* worker = GetWorkerFromInternalField(isolate, args.Holder());
  if (!worker) {
    return;
  }

  worker->Terminate();
}


SerializationData::~SerializationData() {
  // Any ArrayBuffer::Contents are owned by this SerializationData object if
  // ownership hasn't been transferred out via ReadArrayBufferContents.
  // SharedArrayBuffer::Contents may be used by multiple threads, so must be
  // cleaned up by the main thread in CleanupWorkers().
  for (size_t i = 0; i < array_buffer_contents_.size(); ++i) {
    ArrayBuffer::Contents& contents = array_buffer_contents_[i];
    if (contents.Data()) {
      node::node_array_buffer_allocator->Free(contents.Data(),
                                              contents.ByteLength());
    }
  }
}


void SerializationData::WriteTag(SerializationTag tag) { data_.push_back(tag); }


void SerializationData::WriteMemory(const void* p, int length) {
  if (length > 0) {
    int old_size = data_.size();
    for (int i = 0; i < length; i++) data_.push_back(0);
    memcpy(&data_[old_size], p, length);
  }
}


void SerializationData::WriteArrayBufferContents(
    const ArrayBuffer::Contents& contents) {
  array_buffer_contents_.push_back(contents);
  WriteTag(kSerializationTagTransferredArrayBuffer);
  int index = array_buffer_contents_.size() - 1;
  Write(index);
}


void SerializationData::WriteSharedArrayBufferContents(
    const SharedArrayBuffer::Contents& contents) {
  shared_array_buffer_contents_.push_back(contents);
  WriteTag(kSerializationTagTransferredSharedArrayBuffer);
  int index = shared_array_buffer_contents_.size() - 1;
  Write(index);
}


SerializationTag SerializationData::ReadTag(int* offset) const {
  return static_cast<SerializationTag>(Read<uint8_t>(offset));
}


void SerializationData::ReadMemory(void* p, int length, int* offset) const {
  if (length > 0) {
    memcpy(p, &data_[*offset], length);
    (*offset) += length;
  }
}


void SerializationData::ReadArrayBufferContents(ArrayBuffer::Contents* contents,
                                                int* offset) const {
  size_t index = Read<int>(offset);
  CHECK(index < array_buffer_contents_.size());
  *contents = array_buffer_contents_[index];
  // Ownership of this ArrayBuffer::Contents is passed to the caller. Neuter
  // our copy so it won't be double-free'd when this SerializationData is
  // destroyed.
  array_buffer_contents_[index] = ArrayBuffer::Contents();
}


void SerializationData::ReadSharedArrayBufferContents(
    SharedArrayBuffer::Contents* contents, int* offset) const {
  size_t index = Read<int>(offset);
  CHECK(index < shared_array_buffer_contents_.size());
  *contents = shared_array_buffer_contents_[index];
}


void SerializationDataQueue::Enqueue(SerializationData* data) {
  uv_mutex_lock(&mutex_);
  queue_.push(data);
  uv_mutex_unlock(&mutex_);
}


bool SerializationDataQueue::Dequeue(SerializationData** data) {
  uv_mutex_lock(&mutex_);
  *data = NULL;
  if (queue_.empty()) {
    uv_mutex_unlock(&mutex_);
    return false;
  }
  *data = queue_.front();
  queue_.pop();
  uv_mutex_unlock(&mutex_);
  return true;
}


bool SerializationDataQueue::IsEmpty() {
  uv_mutex_lock(&mutex_);
  bool is_empty = queue_.empty();
  uv_mutex_unlock(&mutex_);

  return is_empty;
}


void SerializationDataQueue::Clear() {
  uv_mutex_lock(&mutex_);
  size_t size = queue_.size();
  for (size_t i = 0; i < size; ++i) {
    delete queue_.front();
    queue_.pop();
  }
  uv_mutex_unlock(&mutex_);
}


Worker::Worker(v8::Local<v8::Object> object)
    : thread_(NULL),
      worker_wrapper_(v8::Isolate::GetCurrent(), object),
      script_(NULL),
      script_file_(NULL),
      running_(false) {
  uv_sem_init(&in_semaphore_, 0);
}


Worker::~Worker() {
  if (thread_) {
    delete thread_;
    thread_ = NULL;
  }
  if (script_) {
    delete[] script_;
    script_ = NULL;
  }
  if (script_file_) {
    delete[] script_file_;
    script_file_ = NULL;
  }

  in_message_queue_.Clear();
  out_message_queue_.Clear();
  out_error_queue_.Clear();
  LOG(stdout, "worker destructed!\n");
}


bool Worker::LoadWorkerScript(const v8::FunctionCallbackInfo<v8::Value>& args) {
  String::Utf8Value string(args[0]);
  if (!*string) {
    Throw(args.GetIsolate(), "No parameter for worker constructor!");
    return false;
  }

  bool is_file_name = args.Length() == 1 || args[1]->BooleanValue();
  if (is_file_name) {
    script_file_ = strdup(*string);
    int size;
    if (!(script_ = ReadChars(script_file_, &size))) {
      Throw(args.GetIsolate(), "Read JavaScript file failed!");
      return false;
    }
    return true;
  }

  // The 1st parameter is script, and the 2nd one is 'false';
  script_ = strdup(*string);
  script_file_ = strdup("unnamed");
  return true;
}


void Worker::StartExecuteInThread() {
  running_ = true;
  uv_async_init(uv_default_loop(), &out_message_async_,
                reinterpret_cast<uv_async_cb>(OutMessageCallback));
  uv_async_init(uv_default_loop(), &out_error_async_,
                reinterpret_cast<uv_async_cb>(OutErrorCallback));

  // Shall we keep main thread alive if a worker is alive? we keep currently.
  // uv_unref((uv_handle_t*)&out_message_async_);

  thread_ = new WorkerThread(this);
  thread_->Start();
}


void Worker::PostMessage(SerializationData* data) {
  in_message_queue_.Enqueue(data);
  uv_sem_post(&in_semaphore_);
  LOG(stdout, "A message posted to worker.\n");
}


void Worker::Terminate() {
  CloseCommon(this);

  // To wake up worker if it's waiting for a semaphore
  PostMessage(NULL);
}


void Worker::WaitForThread() {
  Terminate();
  thread_->Join();
}


void Worker::CloseCommon(Worker* worker) {
  NoBarrier_Store(&worker->running_, false);
  uv_unref((uv_handle_t*)&worker->out_message_async_);
  uv_unref((uv_handle_t*)&worker->out_error_async_);
  LOG(stdout, "Worker will be closed\n");
}


void Worker::OutMessageCallback(uv_async_t* async) {
  Isolate* isolate = v8::Isolate::GetCurrent();
  HandleScope handle_scope(isolate);
  Local<Context> context = isolate->GetCurrentContext();
  Environment* env = Environment::GetCurrent(context);
  Worker* worker = ContainerOf(&Worker::out_message_async_, async);

  SerializationData* data = NULL;
  while (worker->out_message_queue_.Dequeue(&data)) {
    if (!data) return;

    int offset = 0;
    Local<Value> data_value;
    if (!DeserializeValue(isolate, *data, &offset).ToLocal(&data_value)) {
      LOG(stdout, "data DeserializeValue is null!\n");
      delete data;
      return;
    }
    delete data;

    Local<Object> event = Object::New(isolate);
    event->Set(env->data_string(), data_value);

    Local<Object> global = context->Global();
    Local<Object> wrap_obj =
        PersistentToLocal(isolate, worker->worker_wrapper_);
    v8::Local<v8::Value> callback = wrap_obj->Get(env->onmessage_string());
    if (callback->IsFunction()) {
      Local<Function> onmessage_fun = Local<Function>::Cast(callback);
      Local<Value> argv[] = {event};
      (void)onmessage_fun->Call(context, global, 1, argv);
    } else {
      LOG(stdout, "onmessage callback is not a function!\n");
    }
  }
}


void Worker::OutErrorCallback(uv_async_t* async) {
  Isolate* isolate = v8::Isolate::GetCurrent();
  HandleScope handle_scope(isolate);
  Local<Context> context = isolate->GetCurrentContext();
  Environment* env = Environment::GetCurrent(context);
  Worker* worker = ContainerOf(&Worker::out_error_async_, async);

  SerializationData* data = NULL;
  while (worker->out_error_queue_.Dequeue(&data)) {
    if (!data) return;

    int offset = 0;
    Local<Value> error_event;
    if (!DeserializeValue(isolate, *data, &offset).ToLocal(&error_event)) {
      LOG(stdout, "data DeserializeValue is null!\n");
      delete data;
      return;
    }
    delete data;

    Local<Object> global = context->Global();
    Local<Object> wrap_obj =
        PersistentToLocal(isolate, worker->worker_wrapper_);
    v8::Local<v8::Value> callback = wrap_obj->Get(env->onerror_string());
    if (callback->IsFunction()) {
      Local<Function> onerror_fun = Local<Function>::Cast(callback);
      Local<Value> argv[] = {error_event};
      (void)onerror_fun->Call(context, global, 1, argv);
    } else {
      LOG(stdout, "onerror callback is not a function!\n");
    }
  }
}


bool Worker::ExecuteString(Isolate* isolate, Local<String> source,
                           Local<String> name) {
  HandleScope handle_scope(isolate);
  TryCatch try_catch(isolate);

  MaybeLocal<Value> maybe_result;
  {
    Local<Script> script = Script::Compile(source, name);
    if (script.IsEmpty()) {
      // Print errors that happened during compilation.
      ReportException(isolate, &try_catch);
      return false;
    }

    maybe_result = script->Run();
  }
  Local<Value> result;
  if (!maybe_result.ToLocal(&result)) {
    CHECK(try_catch.HasCaught());
    // Print errors that happened during execution.
    ReportException(isolate, &try_catch);
    return false;
  }
  CHECK(!try_catch.HasCaught());

  return true;
}


void Worker::ReportException(Isolate* isolate, v8::TryCatch* try_catch) {
  HandleScope handle_scope(isolate);
  Local<Context> utility_context;
  bool enter_context = !isolate->InContext();
  if (enter_context) {
    utility_context = Local<Context>::New(isolate, utility_context_);
    utility_context->Enter();
  }
  v8::String::Utf8Value exception(try_catch->Exception());
  const char* exception_string = ToCString(exception);
  Local<Message> message = try_catch->Message();

  Local<Object> error_message = Object::New(isolate);

  if (message.IsEmpty()) {
    // V8 didn't provide any extra information about this error; just
    // print the exception.
    LOG(stdout, "message is empty!\n");
    LOG(stdout, "%s\n", exception_string);
  } else {
    // Print (filename):(line number): (message).
    v8::String::Utf8Value filename(message->GetScriptOrigin().ResourceName());
    error_message->Set(
        String::NewFromUtf8(isolate, "filename", NewStringType::kNormal)
            .ToLocalChecked(),
        String::NewFromUtf8(isolate, *filename, NewStringType::kNormal)
            .ToLocalChecked());

    const char* filename_string = ToCString(filename);
    int linenum =
        message->GetLineNumber(isolate->GetCurrentContext()).FromJust();
    LOG(stdout, "%s:%i: %s\n", filename_string, linenum, exception_string);
    // Print line of source code.
    v8::String::Utf8Value sourceline(
        message->GetSourceLine(isolate->GetCurrentContext()).ToLocalChecked());
    error_message->Set(
        String::NewFromUtf8(isolate, "lineno", NewStringType::kNormal)
            .ToLocalChecked(),
        Number::New(isolate, linenum));

    const char* sourceline_string = ToCString(sourceline);
    LOG(stdout, "%s\n", sourceline_string);
    // Print wavy underline (GetUnderline is deprecated).
    int start =
        message->GetStartColumn(isolate->GetCurrentContext()).FromJust();
    for (int i = 0; i < start; i++) {
      LOG(stdout, " ");
    }
    int end = message->GetEndColumn(isolate->GetCurrentContext()).FromJust();
    for (int i = start; i < end; i++) {
      LOG(stdout, "^");
    }
    LOG(stdout, "\n");
    error_message->Set(
        String::NewFromUtf8(isolate, "colno", NewStringType::kNormal)
            .ToLocalChecked(),
        Number::New(isolate, start + 1));
    Local<Value> stack_trace_string;
    if (try_catch->StackTrace(isolate->GetCurrentContext())
            .ToLocal(&stack_trace_string) &&
        stack_trace_string->IsString()) {
      v8::String::Utf8Value stack_trace(
          Local<String>::Cast(stack_trace_string));
      LOG(stdout, "%s\n", ToCString(stack_trace));
    }
  }
  LOG(stdout, "\n");
  if (enter_context) utility_context->Exit();


  error_message->Set(
      String::NewFromUtf8(isolate, "message", NewStringType::kNormal)
          .ToLocalChecked(),
      String::NewFromUtf8(isolate, exception_string, NewStringType::kNormal)
          .ToLocalChecked());

  ObjectList to_transfer;
  ObjectList seen_objects;
  SerializationData* data = new SerializationData;
  if (SerializeValue(isolate, error_message, to_transfer, &seen_objects,
                     data)) {
    out_error_queue_.Enqueue(data);
    PerIsolateData* data = PerIsolateData::Get(isolate);
    uv_async_send(&(data->worker_->out_error_async_));
    LOG(stdout, "worker post out a error message!\n");
  }
}


void Worker::ExecuteInThread() {
  Isolate::CreateParams create_params;
  create_params.array_buffer_allocator = node::node_array_buffer_allocator;

  Isolate* isolate = Isolate::New(create_params);
  {
    Isolate::Scope iscope(isolate);
    {
      v8::Locker locker(isolate);
      HandleScope scope(isolate);
      PerIsolateData data(isolate);
      data.worker_ = this;
      Local<Context> context = CreateEvaluationContext(isolate);
      {
        Context::Scope cscope(context);
        Local<Object> global = context->Global();
        InitWorkerGlobalScope(context);
        // First run the script
        Local<String> file_name =
            String::NewFromUtf8(isolate, script_file_, NewStringType::kNormal)
                .ToLocalChecked();

        Local<String> source =
            String::NewFromUtf8(isolate, script_, NewStringType::kNormal)
                .ToLocalChecked();
        if (ExecuteString(isolate, source, file_name)) {
          // Get the message handler
          Local<Value> onmessage =
              global->Get(context, String::NewFromUtf8(isolate, "onmessage",
                                                       NewStringType::kNormal)
                                       .ToLocalChecked())
                  .ToLocalChecked();
          if (onmessage->IsFunction()) {
            Local<Function> onmessage_fun = Local<Function>::Cast(onmessage);
            // Now wait for messages
            while (true && running_) {
              uv_sem_wait(&in_semaphore_);
              SerializationData* data;
              if (!in_message_queue_.Dequeue(&data) || !data) continue;

              int offset = 0;
              Local<Value> data_value;
              if (DeserializeValue(isolate, *data, &offset)
                      .ToLocal(&data_value)) {
                Local<Object> event = Object::New(isolate);
                event->Set(
                    String::NewFromUtf8(isolate, "data", NewStringType::kNormal)
                        .ToLocalChecked(),
                    data_value);
                Local<Value> argv[] = {event};

                TryCatch try_catch(isolate);
                MaybeLocal<Value> maybe_result =
                    onmessage_fun->Call(context, global, 1, argv);
                Local<Value> result;
                if (!maybe_result.ToLocal(&result)) {
                  CHECK(try_catch.HasCaught());
                  // Print errors that happened during execution.
                  ReportException(isolate, &try_catch);
                }
              }
              delete data;
            }
          }
        }
      }
    }
    LOG(stdout, "message loop gone!\n");
    // GC
    isolate->ContextDisposedNotification();
    isolate->IdleNotificationDeadline(
        node::default_platform->MonotonicallyIncreasingTime() + 1.0);

    // By sending a low memory notifications, we will try hard to collect all
    // garbage and will therefore also invoke all weak callbacks of actually
    // unreachable persistent handles.
    isolate->LowMemoryNotification();
  }
  isolate->Dispose();
}


void Worker::InitWorkerGlobalScope(Local<Context>& context) {
  Isolate* isolate = context->GetIsolate();
  Local<Object> global = context->Global();
  Local<Value> this_value = External::New(isolate, this);

  Local<FunctionTemplate> post_message_template =
      FunctionTemplate::New(isolate, PostMessageOut, this_value);

  Local<Function> post_message_function;
  if (post_message_template->GetFunction(context)
          .ToLocal(&post_message_function)) {
    global->Set(context, String::NewFromUtf8(isolate, "postMessage",
                                             NewStringType::kNormal)
                             .ToLocalChecked(),
                post_message_function)
        .FromJust();
  }

  Local<FunctionTemplate> self_close_template =
      FunctionTemplate::New(isolate, SelfClose, this_value);
  Local<Function> self_close_function;
  if (self_close_template->GetFunction(context).ToLocal(&self_close_function)) {
    global->Set(context,
                String::NewFromUtf8(isolate, "close", NewStringType::kNormal)
                    .ToLocalChecked(),
                self_close_function)
        .FromJust();
  }
}


void Worker::PostMessageOut(const v8::FunctionCallbackInfo<v8::Value>& args) {
  Isolate* isolate = args.GetIsolate();
  HandleScope handle_scope(isolate);

  if (args.Length() < 1) {
    Throw(isolate, "Invalid argument");
    return;
  }

  Local<Value> message = args[0];

  // We don't support transfer from worker to master.
  ObjectList to_transfer;

  ObjectList seen_objects;
  SerializationData* data = new SerializationData;
  if (SerializeValue(isolate, message, to_transfer, &seen_objects, data)) {
    CHECK(args.Data()->IsExternal());
    Local<External> this_value = Local<External>::Cast(args.Data());
    Worker* worker = static_cast<Worker*>(this_value->Value());
    worker->out_message_queue_.Enqueue(data);
    PerIsolateData* data = PerIsolateData::Get(isolate);
    uv_async_send(&(data->worker_->out_message_async_));
    LOG(stdout, "worker post out a message!\n");
  } else {
    delete data;
  }
}


void Worker::SelfClose(const v8::FunctionCallbackInfo<v8::Value>& args) {
  Isolate* isolate = args.GetIsolate();
  HandleScope handle_scope(isolate);
  Local<External> this_value = Local<External>::Cast(args.Data());
  Worker* worker = static_cast<Worker*>(this_value->Value());

  CloseCommon(worker);
}


void Worker::WorkerThread::Start() {
  int result;
  result = uv_thread_create(&thread_, &WorkerThread::Run, worker_);
  CHECK_EQ(0, result);
}

void Worker::WorkerThread::Join() { uv_thread_join(&thread_); }

void Worker::WorkerThread::Run(void* arg) {
  Worker* worker = static_cast<Worker*>(arg);

  worker->ExecuteInThread();
}

static bool SerializeArrayBuffer(Isolate* isolate, Local<Value> value,
                                 const ObjectList& to_transfer,
                                 ObjectList* seen_objects,
                                 SerializationData* out_data) {
  Local<ArrayBuffer> array_buffer = Local<ArrayBuffer>::Cast(value);
  if (FindInObjectList(array_buffer, *seen_objects)) {
    Throw(isolate, "Duplicated array buffers not supported");
    return false;
  }
  seen_objects->push_back(array_buffer);
  if (FindInObjectList(array_buffer, to_transfer)) {
    // Transfer ArrayBuffer
    if (!array_buffer->IsNeuterable()) {
      Throw(isolate, "Attempting to transfer an un-neuterable ArrayBuffer");
      return false;
    }

    ArrayBuffer::Contents contents = array_buffer->IsExternal()
                                         ? array_buffer->GetContents()
                                         : array_buffer->Externalize();
    array_buffer->Neuter();
    out_data->WriteArrayBufferContents(contents);
  } else {
    ArrayBuffer::Contents contents = array_buffer->GetContents();
    // Clone ArrayBuffer
    if (contents.ByteLength() > kMaxInt) {
      Throw(isolate, "ArrayBuffer is too big to clone");
      return false;
    }

    int32_t byte_length = static_cast<int32_t>(contents.ByteLength());
    out_data->WriteTag(kSerializationTagArrayBuffer);
    out_data->Write(byte_length);
    out_data->WriteMemory(contents.Data(), byte_length);
  }
  return true;
}

static bool SerializeSharedArrayBuffer(Isolate* isolate, Local<Value> value,
                                       const ObjectList& to_transfer,
                                       ObjectList* seen_objects,
                                       SerializationData* out_data) {
  Local<SharedArrayBuffer> sab = Local<SharedArrayBuffer>::Cast(value);
  if (FindInObjectList(sab, *seen_objects)) {
    Throw(isolate, "Duplicated shared array buffers not supported");
    return false;
  }
  seen_objects->push_back(sab);
  if (!FindInObjectList(sab, to_transfer)) {
    Throw(isolate, "SharedArrayBuffer must be transferred");
    return false;
  }

  SharedArrayBuffer::Contents contents;
  if (sab->IsExternal()) {
    contents = sab->GetContents();
  } else {
    contents = sab->Externalize();
    uv_mutex_lock(&workers_mutex_);
    externalized_shared_contents_.push_back(contents);
    uv_mutex_unlock(&workers_mutex_);
  }
  out_data->WriteSharedArrayBufferContents(contents);
  return true;
}

bool SerializeValue(Isolate* isolate, Local<Value> value,
                    const ObjectList& to_transfer, ObjectList* seen_objects,
                    SerializationData* out_data) {
  CHECK(out_data);
  Local<Context> context = isolate->GetCurrentContext();

  if (value->IsUndefined()) {
    out_data->WriteTag(kSerializationTagUndefined);
  } else if (value->IsNull()) {
    out_data->WriteTag(kSerializationTagNull);
  } else if (value->IsTrue()) {
    out_data->WriteTag(kSerializationTagTrue);
  } else if (value->IsFalse()) {
    out_data->WriteTag(kSerializationTagFalse);
  } else if (value->IsNumber()) {
    Local<Number> num = Local<Number>::Cast(value);
    double value = num->Value();
    out_data->WriteTag(kSerializationTagNumber);
    out_data->Write(value);
  } else if (value->IsString()) {
    v8::String::Utf8Value str(value);
    out_data->WriteTag(kSerializationTagString);
    out_data->Write(str.length());
    out_data->WriteMemory(*str, str.length());
  } else if (value->IsArray()) {
    Local<Array> array = Local<Array>::Cast(value);
    if (FindInObjectList(array, *seen_objects)) {
      Throw(isolate, "Duplicated arrays not supported");
      return false;
    }
    seen_objects->push_back(array);
    out_data->WriteTag(kSerializationTagArray);
    uint32_t length = array->Length();
    out_data->Write(length);
    for (uint32_t i = 0; i < length; ++i) {
      Local<Value> element_value;
      if (array->Get(context, i).ToLocal(&element_value)) {
        if (!SerializeValue(isolate, element_value, to_transfer, seen_objects,
                            out_data))
          return false;
      } else {
        Throw(isolate, "Failed to serialize array element.");
        return false;
      }
    }
  } else if (value->IsArrayBuffer()) {
    SerializeArrayBuffer(isolate, value, to_transfer, seen_objects, out_data);
  } else if (value->IsSharedArrayBuffer()) {
    SerializeSharedArrayBuffer(isolate, value, to_transfer, seen_objects,
                               out_data);
  } else if (value->IsArrayBufferView()) {
    Local<ArrayBufferView> array_buffer_view =
        Local<ArrayBufferView>::Cast(value);

    // Serialize BufferView
    if (array_buffer_view->IsInt8Array()) {
      out_data->WriteTag(kSerializationTagByteArray);
    } else if (array_buffer_view->IsUint8Array()) {
      out_data->WriteTag(kSerializationTagUnsignedByteArray);

    } else if (array_buffer_view->IsUint8ClampedArray()) {
      out_data->WriteTag(kSerializationTagUnsignedByteClampedArray);

    } else if (array_buffer_view->IsInt16Array()) {
      out_data->WriteTag(kSerializationTagShortArray);

    } else if (array_buffer_view->IsUint16Array()) {
      out_data->WriteTag(kSerializationTagUnsignedShortArray);

    } else if (array_buffer_view->IsInt32Array()) {
      out_data->WriteTag(kSerializationTagIntArray);

    } else if (array_buffer_view->IsUint32Array()) {
      out_data->WriteTag(kSerializationTagUnsignedIntArray);

    } else if (array_buffer_view->IsFloat32Array()) {
      out_data->WriteTag(kSerializationTagFloatArray);

    } else if (array_buffer_view->IsFloat64Array()) {
      out_data->WriteTag(kSerializationTagDoubleArray);
    }

    out_data->Write(array_buffer_view->ByteOffset());
    out_data->Write(array_buffer_view->ByteLength());

    // Serialize buffer
    Local<Value> buffer = array_buffer_view->Buffer().As<Value>();
    if (buffer->IsArrayBuffer())
      SerializeArrayBuffer(isolate, buffer, to_transfer, seen_objects,
                           out_data);
    else
      SerializeSharedArrayBuffer(isolate, buffer, to_transfer, seen_objects,
                                 out_data);

  } else if (value->IsObject()) {
    Local<Object> object = Local<Object>::Cast(value);
    if (FindInObjectList(object, *seen_objects)) {
      Throw(isolate, "Duplicated objects not supported");
      return false;
    }
    seen_objects->push_back(object);
    Local<Array> property_names;
    if (!object->GetOwnPropertyNames(context).ToLocal(&property_names)) {
      Throw(isolate, "Unable to get property names");
      return false;
    }

    uint32_t length = property_names->Length();
    out_data->WriteTag(kSerializationTagObject);
    out_data->Write(length);
    for (uint32_t i = 0; i < length; ++i) {
      Local<Value> name;
      Local<Value> property_value;
      if (property_names->Get(context, i).ToLocal(&name) &&
          object->Get(context, name).ToLocal(&property_value)) {
        if (!SerializeValue(isolate, name, to_transfer, seen_objects, out_data))
          return false;
        if (!SerializeValue(isolate, property_value, to_transfer, seen_objects,
                            out_data))
          return false;
      } else {
        Throw(isolate, "Failed to serialize property.");
        return false;
      }
    }
  } else {
    Throw(isolate, "Don't know how to serialize object");
    return false;
  }

  return true;
}

#define DESERIALIZE_TYPED_ARRAY(Type, ElementSize)                          \
  do {                                                                      \
    size_t byte_offset = data.Read<size_t>(offset);                         \
    size_t byte_length = data.Read<size_t>(offset);                         \
    Local<Value> array_buffer;                                              \
    CHECK(DeserializeValue(isolate, data, offset).ToLocal(&array_buffer));  \
    if (array_buffer->IsArrayBuffer())                                      \
      result = Type::New(array_buffer.As<ArrayBuffer>(), byte_offset,       \
                         byte_length / ElementSize);                        \
    else                                                                    \
      result = Type::New(array_buffer.As<SharedArrayBuffer>(), byte_offset, \
                         byte_length / ElementSize);                        \
  } while (0)

MaybeLocal<Value> DeserializeValue(Isolate* isolate,
                                   const SerializationData& data, int* offset) {
  CHECK(offset);
  EscapableHandleScope scope(isolate);

  Local<Value> result;
  SerializationTag tag = data.ReadTag(offset);

  switch (tag) {
    case kSerializationTagUndefined:
      result = Undefined(isolate);
      break;
    case kSerializationTagNull:
      result = Null(isolate);
      break;
    case kSerializationTagTrue:
      result = True(isolate);
      break;
    case kSerializationTagFalse:
      result = False(isolate);
      break;
    case kSerializationTagNumber:
      result = Number::New(isolate, data.Read<double>(offset));
      break;
    case kSerializationTagString: {
      int length = data.Read<int>(offset);
      CHECK(length >= 0);
      std::vector<char> buffer(length + 1);  // + 1 so it is never empty.
      data.ReadMemory(&buffer[0], length, offset);
      MaybeLocal<String> str =
          String::NewFromUtf8(isolate, &buffer[0], NewStringType::kNormal,
                              length)
              .ToLocalChecked();
      if (!str.IsEmpty()) result = str.ToLocalChecked();
      break;
    }
    case kSerializationTagArray: {
      uint32_t length = data.Read<uint32_t>(offset);
      Local<Array> array = Array::New(isolate, length);
      for (uint32_t i = 0; i < length; ++i) {
        Local<Value> element_value;
        CHECK(DeserializeValue(isolate, data, offset).ToLocal(&element_value));
        array->Set(isolate->GetCurrentContext(), i, element_value).FromJust();
      }
      result = array;
      break;
    }
    case kSerializationTagObject: {
      int length = data.Read<int>(offset);
      Local<Object> object = Object::New(isolate);
      for (int i = 0; i < length; ++i) {
        Local<Value> property_name;
        CHECK(DeserializeValue(isolate, data, offset).ToLocal(&property_name));
        Local<Value> property_value;
        CHECK(DeserializeValue(isolate, data, offset).ToLocal(&property_value));
        object->Set(isolate->GetCurrentContext(), property_name, property_value)
            .FromJust();
      }
      result = object;
      break;
    }
    case kSerializationTagArrayBuffer: {
      int32_t byte_length = data.Read<int32_t>(offset);
      Local<ArrayBuffer> array_buffer = ArrayBuffer::New(isolate, byte_length);
      ArrayBuffer::Contents contents = array_buffer->GetContents();
      CHECK(static_cast<size_t>(byte_length) == contents.ByteLength());
      data.ReadMemory(contents.Data(), byte_length, offset);
      result = array_buffer;
      break;
    }
    case kSerializationTagTransferredArrayBuffer: {
      ArrayBuffer::Contents contents;
      data.ReadArrayBufferContents(&contents, offset);
      result = ArrayBuffer::New(isolate, contents.Data(), contents.ByteLength(),
                                ArrayBufferCreationMode::kInternalized);
      break;
    }
    case kSerializationTagTransferredSharedArrayBuffer: {
      SharedArrayBuffer::Contents contents;
      data.ReadSharedArrayBufferContents(&contents, offset);
      result = SharedArrayBuffer::New(isolate, contents.Data(),
                                      contents.ByteLength());
      break;
    }
    case kSerializationTagByteArray: {
      DESERIALIZE_TYPED_ARRAY(Int8Array, 1);
      break;
    }
    case kSerializationTagUnsignedByteArray: {
      DESERIALIZE_TYPED_ARRAY(Uint8Array, 1);
      break;
    }
    case kSerializationTagUnsignedByteClampedArray: {
      DESERIALIZE_TYPED_ARRAY(Uint8ClampedArray, 1);
      break;
    }
    case kSerializationTagShortArray: {
      DESERIALIZE_TYPED_ARRAY(Int16Array, 2);
      break;
    }
    case kSerializationTagUnsignedShortArray: {
      DESERIALIZE_TYPED_ARRAY(Uint16Array, 2);
      break;
    }
    case kSerializationTagIntArray: {
      DESERIALIZE_TYPED_ARRAY(Int32Array, 4);
      break;
    }
    case kSerializationTagUnsignedIntArray: {
      DESERIALIZE_TYPED_ARRAY(Uint32Array, 4);
      break;
    }
    case kSerializationTagFloatArray: {
      DESERIALIZE_TYPED_ARRAY(Float32Array, 4);
      break;
    }
    case kSerializationTagDoubleArray: {
      DESERIALIZE_TYPED_ARRAY(Float64Array, 8);
      break;
    }
    default:
      UNREACHABLE();
  }

  return scope.Escape(result);
}

#undef DESERIALIZE_TYPED_ARRAY


void CleanupWorkers() {
  // Make a copy of workers_, because we don't want to call Worker::Terminate
  // while holding the workers_mutex_ lock. Otherwise, if a worker is about to
  // create a new Worker, it would deadlock.
  std::vector<Worker*> workers_copy;
  {
    uv_mutex_lock(&workers_mutex_);
    allow_new_workers_ = false;
    workers_copy.swap(workers_);
    uv_mutex_unlock(&workers_mutex_);
  }

  for (size_t i = 0; i < workers_copy.size(); ++i) {
    Worker* worker = workers_copy[i];
    worker->WaitForThread();
    delete worker;
  }

  // Now that all workers are terminated, we can re-enable Worker creation.
  uv_mutex_lock(&workers_mutex_);
  allow_new_workers_ = true;

  for (size_t i = 0; i < externalized_shared_contents_.size(); ++i) {
    const SharedArrayBuffer::Contents& contents =
        externalized_shared_contents_[i];
    node::node_array_buffer_allocator->Free(contents.Data(),
                                            contents.ByteLength());
  }
  externalized_shared_contents_.clear();
  uv_mutex_unlock(&workers_mutex_);
}


char* ReadChars(const char* name, int* size_out) {
  FILE* file = fopen(name, "rb");
  if (file == NULL) return NULL;

  fseek(file, 0, SEEK_END);
  size_t size = ftell(file);
  rewind(file);

  char* chars = new char[size + 1];
  chars[size] = '\0';
  for (size_t i = 0; i < size;) {
    i += fread(&chars[i], 1, size - i, file);
    if (ferror(file)) {
      fclose(file);
      delete[] chars;
      return nullptr;
    }
  }
  fclose(file);
  *size_out = static_cast<int>(size);
  return chars;
}


static void Print(const v8::FunctionCallbackInfo<v8::Value>& args) {
  for (int i = 0; i < args.Length(); i++) {
    HandleScope handle_scope(args.GetIsolate());
    if (i != 0) {
      printf(" ");
    }

    // Explicitly catch potential exceptions in toString().
    v8::TryCatch try_catch(args.GetIsolate());
    Local<String> str_obj;
    if (!args[i]
             ->ToString(args.GetIsolate()->GetCurrentContext())
             .ToLocal(&str_obj)) {
      try_catch.ReThrow();
      return;
    }

    v8::String::Utf8Value str(str_obj);
    int n = static_cast<int>(fwrite(*str, sizeof(**str), str.length(), stdout));
    if (n != str.length()) {
      printf("Error in fwrite\n");
      fflush(stdout);
      fflush(stderr);
      _exit(1);
    }
  }
  printf("\n");
  fflush(stdout);
}

Handle<ObjectTemplate> CreateGlobalTemplate(Isolate* isolate) {
  Local<ObjectTemplate> global_template = ObjectTemplate::New(isolate);

  global_template->Set(
      String::NewFromUtf8(isolate, "print", NewStringType::kNormal)
          .ToLocalChecked(),
      FunctionTemplate::New(isolate, Print));

  return global_template;
}

void InitWorker(Local<Object> target, Local<Value> unused,
                Local<Context> context) {
  Environment* env = Environment::GetCurrent(context);

  Local<FunctionTemplate> t = env->NewFunctionTemplate(WorkerNew);
  t->SetClassName(FIXED_ONE_BYTE_STRING(env->isolate(), "Worker"));
  t->InstanceTemplate()->SetInternalFieldCount(1);
  t->ReadOnlyPrototype();

  env->SetProtoMethod(t, "terminate", WorkerTerminate);
  env->SetProtoMethod(t, "postMessage", WorkerPostMessage);

  env->set_worker_constructor_template(t);
  target->Set(FIXED_ONE_BYTE_STRING(env->isolate(), "Worker"),
              t->GetFunction());
}

}  // namespace Worker
}  // namespace node

NODE_MODULE_CONTEXT_AWARE_BUILTIN(worker, node::Worker::InitWorker)
