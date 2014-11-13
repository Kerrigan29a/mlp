CPPFLAGS	?= -DDEBUG
CFLAGS		?= -Wall -Werror

all: mlp README.md

mlp.o: mlp.c

mlp.c: mlp.nw
	./noweb.py/noweb.py -o mlp.c tangle -R mlp.c mlp.nw

README.md: mlp.nw
	./noweb.py/noweb.py -o README.md weave --default-code-syntax=c mlp.nw

clean:
	-@$(RM) mlp mlp.o

distclean: clean
	-@$(RM) mlp.c README.md
