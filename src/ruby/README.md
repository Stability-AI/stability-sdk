# Ruby

This guide will walk you through generating the Ruby bindings using Protobuf and gRPC to make requests to the Stability AI API.


## Installation

Add the following gem dependencies to your project:

```ruby
gem "google-protobuf"
gem "grpc"
```

You will also need to make use of the `grpc-tools` gem for code generation, but only once. So this gem is not actually a project dependency:

```bash
gem install grpc-tools
```

For platform specific help on installing the gem dependencies, see the docs for the gems:

* https://github.com/protocolbuffers/protobuf/tree/main/ruby
* https://github.com/grpc/grpc/blob/master/src/ruby/README.md


## Code generation

Fetch the [`generation.proto`](https://github.com/Stability-AI/stability-sdk/tree/ecma_clients/src/proto) file and add it to your project.

You will use the `grpc_tools_ruby_protoc` tool from `grpc-tools` to generate the Protobuf messages and gRPC service class. This only needs to be done once, unless the `.proto` file changes:

```bash
grpc_tools_ruby_protoc -I ./protobuf --ruby_out=./lib --grpc_out=./lib protobuf/generation.proto
```

Modify the paths here as required to match the structure of your project.

This will generate two files which you should then require into your project, `generation_pb.rb` and `generation_services_pb.rb`.


## Usage

Instantiate a service stub class using the API key from the web interface:

```ruby
channel_credentials = GRPC::Core::ChannelCredentials.new()
call_credentials = GRPC::Core::CallCredentials.new(->(_) { { authorization: "Bearer #{ api_key }" } })
stub = Gooseai::GenerationService::Stub.new("grpc.stability.ai:443", channel_credentials.compose(call_credentials))
```

Construct your `Request` object to suit your needs. Most of these fields are optional and can be omitted. The [python client](https://github.com/Stability-AI/stability-sdk/blob/main/src/stability_sdk/client.py) provides some good inspiraton for further usage beyond what is shown here:

```ruby
request = Gooseai::Request.new(
  requested_type: Gooseai::ArtifactType::ARTIFACT_IMAGE,
  engine_id: "stable-diffusion-v1-5",
  request_id: SecureRandom.uuid,
  classifier: Gooseai::ClassifierParameters.new(),
  prompt: [
    Gooseai::Prompt.new(
      text: "TODO: your text prompt here"
    )
  ],
  image: Gooseai::ImageParameters.new(
    height: 512,
    width: 512,
    steps: 50,
    samples: 1,
    seed: [ rand(4294967295) ],
    transform: Gooseai::TransformType.new(diffusion: Gooseai::DiffusionSampler::SAMPLER_K_LMS),
    parameters: [
      Gooseai::StepParameter.new(
        scaled_step: 0,
        sampler: Gooseai::SamplerParameters.new(cfg_scale: 7.0)
      ),
    ],
  )
)
```

Finally, make the gRPC request. The call to `generate` returns an iterator that streams `Answer` messages from the server. An `Answer` with no `artifacts` is a keepalive heartbeat:

```ruby
answers = []

stub.generate(request).each do |answer|
  if answer.artifacts.any?
    answers << answer
  else
    # keepalive answer with no artifacts
  end
end
```

Image artifacts can then be extracted which contain binary data:
```ruby
artifacts = answers.map do |answer|
  answer.artifacts.select do |artifact|
    artifact.type == :ARTIFACT_IMAGE
  end
end.flatten
```
