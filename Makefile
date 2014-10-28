CPPFLAGS	?= -DDEBUG
CFLAGS		?= -Wall -Werror

all: mlp

mlp.o: mlp.c

clean:
	-@$(RM) mlp mlp.o

distclean: clean
