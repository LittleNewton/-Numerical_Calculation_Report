PROJECT_NAME = 02_非线性方程求根

main.pdf : *.tex Makefile section/*.tex ../style/*.tex
	xelatex --synctex=-1 main.tex
	#biber main
	#bibtex chapter/*.aux
	#xelatex --synctex=-1 main.tex
	#xelatex --synctex=-1 main.tex
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
	#-rm main.pdf