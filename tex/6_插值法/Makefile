STUDENT_NAME = 20151910042-刘鹏
PROJECT_NAME = ${STUDENT_NAME}-6-插值法

main.pdf : *.tex Makefile ../style/*.tex
	xelatex --synctex=-1 main.tex
	biber main
	xelatex --synctex=-1 main.tex
	xelatex --synctex=-1 main.tex
	mv main.pdf ../../build/${PROJECT_NAME}.pdf
	make clean

clean:
	-rm *.aux
	-rm style/*.aux
	-rm *.aux *.out *.log *.bbl *.toc *.blg
	-rm *.bbl
	-rm *.blg
	-rm section/*.bbl
	-rm section/*.blg
	-rm section/*.aux
	-rm *.bcf
	-rm *.xml
	-rm *.synctex
	-rm main.pdf