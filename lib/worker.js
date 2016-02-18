'use strict';

const binding = process.binding('worker');
global.Worker = binding.Worker;
