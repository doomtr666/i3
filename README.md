# Welcome to I3

Years ago, my team and I released a game on Steam: https://store.steampowered.com/app/599040/?snr=1_5_9__205.
Unfortunately, we were unable to continue development. For that game, I wrote a complete game engine called "Insanity Engine" in C#, which included a DirectX 11 renderer, collision and physics systems, network code, server infrastructure, and more.
I3 is my modest third attempt—mainly to explore new technologies—but who knows what the future holds :)

This project is a work in progress and experimental; for now, it is intended only for educational purposes.
It features a Vulkan-based renderer, with possible support for DirectX 12 in the future.
The codebase is a mix of C and C#.

# Requirements

- [Install Bazelisk](https://github.com/bazelbuild/bazelisk#installation) (recommended Bazel launcher)
- [Install Bazel for Windows](https://bazel.build/install/windows)

# run tests

```
bazelisk test //native/core_tst
```

# Build and run

``` 
bazelisk run //samples/vk_draw_cubes
```
