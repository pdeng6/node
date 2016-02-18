#ifndef SRC_NODE_WORKER_H_
#define SRC_NODE_WORKER_H_

#include "node.h"
#include "v8.h"

#include "uv.h"

#include <vector>
#include <queue>

namespace node {
namespace Worker {

using v8::ArrayBuffer;
using v8::Context;
using v8::FunctionCallbackInfo;
using v8::Isolate;
using v8::Local;
using v8::MaybeLocal;
using v8::Object;
using v8::SharedArrayBuffer;
using v8::String;
using v8::TryCatch;
using v8::Value;

enum SerializationTag {
  kSerializationTagUndefined,
  kSerializationTagNull,
  kSerializationTagTrue,
  kSerializationTagFalse,
  kSerializationTagNumber,
  kSerializationTagString,
  kSerializationTagArray,
  kSerializationTagObject,
  kSerializationTagArrayBuffer,
  kSerializationTagTransferredArrayBuffer,
  kSerializationTagTransferredSharedArrayBuffer,
  kSerializationTagByteArray,
  kSerializationTagUnsignedByteArray,
  kSerializationTagUnsignedByteClampedArray,
  kSerializationTagShortArray,
  kSerializationTagUnsignedShortArray,
  kSerializationTagIntArray,
  kSerializationTagUnsignedIntArray,
  kSerializationTagFloatArray,
  kSerializationTagDoubleArray,
};

class SerializationData {
 public:
  SerializationData() {}
  ~SerializationData();

  void WriteTag(SerializationTag tag);
  void WriteMemory(const void* p, int length);
  void WriteArrayBufferContents(const ArrayBuffer::Contents& contents);
  void WriteSharedArrayBufferContents(
      const SharedArrayBuffer::Contents& contents);

  template <typename T>
  void Write(const T& data) {
    WriteMemory(&data, sizeof(data));
  }

  SerializationTag ReadTag(int* offset) const;
  void ReadMemory(void* p, int length, int* offset) const;
  void ReadArrayBufferContents(ArrayBuffer::Contents* contents,
                               int* offset) const;
  void ReadSharedArrayBufferContents(SharedArrayBuffer::Contents* contents,
                                     int* offset) const;

  template <typename T>
  T Read(int* offset) const {
    T value;
    ReadMemory(&value, sizeof(value), offset);
    return value;
  }

 private:
  std::vector<uint8_t> data_;
  mutable std::vector<ArrayBuffer::Contents> array_buffer_contents_;
  std::vector<SharedArrayBuffer::Contents> shared_array_buffer_contents_;
};


class SerializationDataQueue {
 public:
  SerializationDataQueue() { uv_mutex_init(&mutex_); }
  void Enqueue(SerializationData* data);
  bool Dequeue(SerializationData** data);
  bool IsEmpty();
  void Clear();
  ~SerializationDataQueue() { uv_mutex_destroy(&mutex_); }

 private:
  uv_mutex_t mutex_;
  std::queue<SerializationData*> queue_;
};


class Worker {
 public:
  Worker(v8::Local<v8::Object> object);
  ~Worker();

  bool LoadWorkerScript(const v8::FunctionCallbackInfo<v8::Value>& args);

  // Run the given script on this Worker. This function should only be called
  // once, and should only be called by the thread that created the Worker.
  void StartExecuteInThread();
  // Post a message to the worker's incoming message queue. The worker will
  // take ownership of the SerializationData.
  // This function should only be called by the thread that created the Worker.
  void PostMessage(SerializationData* data);

  // Terminate the worker's event loop. Messages from the worker that have been
  // queued can still be read via GetMessage().
  // This function can be called by any thread.
  void Terminate();
  // Terminate and join the thread.
  // This function can be called by any thread.
  void WaitForThread();

  bool IsRunning() { return running_; }

 private:
  class WorkerThread {
   public:
    explicit WorkerThread(Worker* worker) : worker_(worker) {}

    void Join();
    void Start();

   private:
    static void Run(void* arg);
    Worker* worker_;
    uv_thread_t thread_;
  };

  // Method should be called on any thread.
  static void CloseCommon(Worker* worker);

  // Methods should be called on the thread that created the worker.
  static void OutMessageCallback(uv_async_t*);
  static void OutErrorCallback(uv_async_t* async);

  // Methods should be called on worker thread.
  bool ExecuteString(Isolate* isolate, Local<String> source,
                     Local<String> name);
  void ReportException(Isolate* isolate, TryCatch* try_catch);
  void ExecuteInThread();
  void InitWorkerGlobalScope(Local<Context>& context);
  static void PostMessageOut(const v8::FunctionCallbackInfo<v8::Value>& args);
  static void SelfClose(const v8::FunctionCallbackInfo<v8::Value>& args);

  uv_async_t out_message_async_;
  uv_async_t out_error_async_;
  uv_sem_t in_semaphore_;
  SerializationDataQueue in_message_queue_;
  SerializationDataQueue out_message_queue_;
  SerializationDataQueue out_error_queue_;
  WorkerThread* thread_;
  v8::Persistent<v8::Object> worker_wrapper_;
  char* script_;
  char* script_file_;
  int32_t running_;
};

void CleanupWorkers();

}  // namespace Worker
}  // namespace node

#endif  // SRC_NODE_WORKER_H_
