# Cortex

Deep learning library developed in C.

## Compile

Compile the library with

```bash
make
```

## Install

Install the library with

```bash
sudo make install
```

This will install the library in `/usr/local/lib` and move all header files to `/usr/local/include/cortex`.

## Uninstall

You can uninstall the library by running

```bash
sudo make uninstall
```

## Include and Run the Library

Inlcude the library using

```C
#include <cortex.h>
```

and compile your application with

```bash
gcc -o my_program my_program.c -lcortex -L/usr/local/lib -I/usr/local/include/cortex
```

Then, you need to make sure that `/usr/local/lib` is always included in the shared library search path. To do this, run

```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

and

```bash
source ~/.bashrc
```

Finally, run your program with

```bash
./my_program
```