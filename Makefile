# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -Iinclude -fPIC -O3 -fopenmp -ffast-math

# Directories
SRCDIR = src
OBJDIR = build/obj
INCDIR = include
LIBDIR = build/lib
BINDIR = build/bin

# Installation paths (can be overridden by user)
PREFIX ?= /usr/local
INCLUDE_INSTALL_DIR ?= $(PREFIX)/include/cortex
LIB_INSTALL_DIR ?= $(PREFIX)/lib

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
	$(CC) $(CFLAGS) -c $< -o $@ -lm

# Create the shared library
$(SHARED_LIB): $(OBJ) | $(LIBDIR)
	$(CC) -shared -o $(SHARED_LIB) $(OBJ) -lm

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
	@mkdir -p $(LIB_INSTALL_DIR) $(INCLUDE_INSTALL_DIR)
	@cp $(SHARED_LIB) $(LIB_INSTALL_DIR)/
	@cp -r $(INCDIR)/* $(INCLUDE_INSTALL_DIR)/
	@echo "Shared library installed to $(LIB_INSTALL_DIR)"
	@echo "Headers installed to $(INCLUDE_INSTALL_DIR)"

# Uninstall the library and headers
uninstall:
	rm -f $(LIB_INSTALL_DIR)/libcortex.so
	rm -rf $(INCLUDE_INSTALL_DIR)
	@echo "Shared library and headers uninstalled"