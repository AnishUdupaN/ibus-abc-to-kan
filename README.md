prerequisites:
podman
linux(tested on fedora 44 workstation)

how to use:
run setup_host.sh and it will do it everything itself. You just have to type out the sudo password once.

note:
I used containers because I did not want to have the trouble of finding where are the model caches, config, runtimes, that we do not require at finish and deleting them one by one. as this does not need any dependencies like torch or cuda at runtime because of the requirement of this to be fast on startup, it has to start within 2 seconds of program start, and uses very less memory, I think less than 500mb ram, and in my testing, it peaked somewhere at 5% cpu usage in my laptop when typing fast. I think that is very much less now in this final version. and the model is just around 550mb.
in the previous versions, it had to lookup through around 1.2 million words, which required high cpu usage, but now it uses sqlite with many optimizations and also with some caching for fetching recently used words fast. the latency is about 20-60ms for individual words and about 0.01ms for recently used words.
