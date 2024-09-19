# {C}ortex

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

This will install the library in `/usr/local/lib` and copy all header files to `/usr/local/include/cortex` by default. You can also specify a custom installation path using

```bash
sudo make install PREFIX=/installation/path
```

## Uninstall

You can uninstall the library by running

```bash
sudo make uninstall
```

or, in case of custom installation path

```bash
sudo make uninstall PREFIX=/installation/path
```

## Include and Run the Library

Inlcude the library using

```C
#include <cortex.h>
```

and compile your application with

```bash
gcc -o my_program my_program.c -lcortex -L/installation/path/lib -I/installation/path/include/cortex -lm
```

Then, you need to make sure that the installation path is always included in the shared library search path. To do this, run

```bash
export LD_LIBRARY_PATH=/installation/path/lib:$LD_LIBRARY_PATH
```

and

```bash
source ~/.bashrc
```

Finally, run your program with

```bash
./my_program
```