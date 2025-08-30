INFO:     Started server process [34792]
INFO:     Waiting for application startup.
INFO:     Application starting... Loading models.
Some weights of LlamaForCausalLM were not initialized from the model checkpoint at ighoshsubho/Bitnet-SmolLM-135M and are newly initialized: ['model.layers.0.input_layernorm.weight', 'model.layers.1.input_layernorm.weight', 'model.layers.2.input_layernorm.weight', 'model.layers.3.input_layernorm.weight', 'model.layers.4.input_layernorm.weight', 'model.layers.5.input_layernorm.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
INFO:     Successfully loaded model: 'ighoshsubho/Bitnet-SmolLM-135M'
INFO:     Model loading time: 1.84 seconds
INFO:     Successfully loaded embedding model: 'sentence-transformers/all-MiniLM-L6-v2'
INFO:     Total model loading time: 4.45 seconds
INFO:     Application startup complete.
INFO:     Starting real-time web crawl for query: 'What is a Kubernetes Pod? Answer short'
INFO:     Crawling: https://kubernetes.io/docs/concepts/overview/
INFO:     Crawling: https://kubernetes.io/docs/concepts/workloads/pods/
INFO:     Crawling: https://kubernetes.io/docs/concepts/services-networking/service/
INFO:     Crawling: https://kubernetes.io/docs/tasks/debug/debug-application/debug-pods/
INFO:     Crawling: https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/
INFO:     Crawling: https://prometheus.io/docs/guides/node-exporter/
INFO:     Crawling: https://docs.spring.io/spring-boot/docs/current/reference/html/actuator.html
INFO:     Crawling finished. Found 57 text chunks.
INFO:     Context retrieval took 7.64 seconds.
The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
This is a friendly reminder - the current text generation call will exceed the model's predefined maximum length (768). Depending on the model, you may observe exceptions, performance degradation, or nothing at all.
INFO:     Total processing time: 14.43 seconds.
INFO:     127.0.0.1:59402 - "POST /generate HTTP/1.1" 200 OK

#####################
$ curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is a Kubernetes Pod?"}'

{
   "response":"### 1:\n\n\nIn computer programming and web applications, let's's first example for the following process:\n\n### 2:\n\n\nIn computer programming and web programming, let's's first step to the following process:\n\n### *\n\nThe following unit unit is a highly popular tool for computer programming and web developers. These two-dimensional system is a highly popular tool for computer programming and web development. It's a simple piece of this method to the following unit, including its core, which is a fundamental aspect of computer programming and web programming and web development.\n\n**Section:\n\nIn computer programming and web development, let's's approach to the following unit:\n\n### Introduction\n\nIn today's digital world, let's first first-order software software is a digital tool designed to enable users to search and web development. It's a robust platform for this system, including the following process, which is a highly popular tool",
   "retrieved_context_summary":"Pods Pods are the smallest deployable units of computing that you can create and manage in Kubernetes. A Pod (as in a pod of whales or pea pod) is a group of one or more containers , with shared stora...",
   "debug_info":{
      "total_chunks_found":57,
      "retrieval_time_seconds":7.38,
      "total_time_seconds":14.12
   }
}
