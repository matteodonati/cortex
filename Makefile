# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -Iinclude

# Directories
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Source files
SRC = $(shell find $(SRCDIR) -name '*.c')
OBJ = $(patsubst $(SRCDIR)/%, $(OBJDIR)/%, $(SRC:.c=.o))

# Target executable
TARGET = $(BINDIR)/cortex

# Default rule
all: $(TARGET)

# Ensure obj directory exists
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Ensure bin directory exists
$(BINDIR):
	mkdir -p $(BINDIR)

# Create the directories needed for object files
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Linking
$(TARGET): $(OBJ) | $(BINDIR)
	$(CC) $(CFLAGS) $(OBJ) -o $(TARGET)

# Clean up
clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Print SRC
print-src:
	@echo "SRC = $(SRC)"

# Print OBJ
print-obj:
	@echo "OBJ = $(OBJ)"
