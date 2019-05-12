default: generate

TASK2 = tex/2_非线性方程求根/
TASK3 = tex/3_线性方程组的直接解法/
TASK2_DETAIL = ${TASK2}section/*.tex/
ALL_OBJ = 

generate : ${TASK2}*.tex/ Makefile ${TASK2}Makefile
	#make clean
	-mkdir build
	cd ${TASK2} && make && make clean
	cd ${TASK3} && make && make clean

clean:
	cd ${TASK2} && make clean
	cd ${TASK3} && make clean