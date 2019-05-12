main.pdf : *.tex Makefile
	xelatex --synctex=-1 main.tex
	#biber main
	#bibtex chapter/*.aux
	#xelatex --synctex=-1 main.tex
	#xelatex --synctex=-1 main.tex
	#mv main.pdf ../build/20151910042-刘鹏-毕业论文.pdf
	make clean

clean:
	-rm *.aux
	-rm style/*.aux
	-rm *.aux *.out *.log *.bbl *.toc *.blg
	-rm *.bbl
	-rm *.blg
	-rm chapter/*.bbl
	-rm chapter/*.blg
	-rm *.bcf
	-rm *.xml
	-rm *.synctex
	#-rm main.pdf