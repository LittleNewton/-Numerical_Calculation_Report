default: generate

generate: tex/*.tex tex/reference.bib tex/Makefile Makefile
	make clean
	-mkdir build
	cd tex && make && make clean

clean:
	cd tex && make clean
