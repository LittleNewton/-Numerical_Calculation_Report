default: generate

TASK2 = tex/2_非线性方程求根/
TASK2_DETAIL = ${TASK2}section/*.tex/

generate : ${TASK2}section/*.tex/ Makefile ${TASK2}*.tex ${TASK2}Makefile
	#make clean
	-mkdir build
	cd ${TASK2} && make && make clean

clean:
	cd tex && make clean
