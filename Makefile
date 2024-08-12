# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -Iinclude -fPIC

# Directories
SRCDIR = src
OBJDIR = build/obj
INCDIR = include
LIBDIR = build/lib
BINDIR = build/bin
PREFIX = /usr/local

# Source files and object files
SRC = $(shell find $(SRCDIR) -name '*.c')
OBJ = $(patsubst $(SRCDIR)/%, $(OBJDIR)/%, $(SRC:.c=.o))

# Shared library target
SHARED_LIB = $(LIBDIR)/libcortex.so

# Default rule
all: $(SHARED_LIB)

# Ensure necessary directories exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(LIBDIR):
	mkdir -p $(LIBDIR)

# Compile source files into object files
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Create the shared library
$(SHARED_LIB): $(OBJ) | $(LIBDIR)
	$(CC) -shared -o $(SHARED_LIB) $(OBJ)

# Clean up
clean:
	rm -rf $(OBJDIR) $(LIBDIR) ./build

# Print SRC
print-src:
	@echo "SRC = $(SRC)"

# Print OBJ
print-obj:
	@echo "OBJ = $(OBJ)"

# Install the library and headers
install: $(SHARED_LIB)
	@mkdir -p $(PREFIX)/lib $(PREFIX)/include/cortex
	@cp $(SHARED_LIB) $(PREFIX)/lib/
	@cp -r $(INCDIR)/* $(PREFIX)/include/cortex/
	@echo "Shared library installed to $(PREFIX)/lib"
	@echo "Headers installed to $(PREFIX)/include/cortex"

# Uninstall the library and headers
uninstall:
	rm -f $(PREFIX)/lib/libcortex.so
	rm -rf $(PREFIX)/include/cortex
	@echo "Shared library and headers uninstalled"
