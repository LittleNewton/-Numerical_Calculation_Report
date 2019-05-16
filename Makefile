default: generate

TASK2 = tex/2_非线性方程求根/
TASK3 = tex/3_线性方程组的直接解法/
TASK4 = tex/4_解线性方程组的迭代法/
TASK6 = tex/6_插值法/

TEX_FILE = ${TASK2}*.tex ${TASK3}*.tex ${TASK4}*.tex ${TASK6}*.tex
SUB_MAKEFILE = ${TASK2}Makefile ${TASK3}Makefile ${TASK4}Makefile ${TASK6}Makefile
ALL_OBJ = ${TEX_FILE} ${SUB_MAKEFILE}

task2 : ${TASK2}*.tex/ Makefile ${TASK2}Makefile
	cd ${TASK2} && make && make clean

task3 : ${TASK3}*.tex/ Makefile ${TASK3}Makefile
	cd ${TASK3} && make && make clean

task4 : ${TASK4}*.tex/ Makefile ${TASK4}Makefile
	cd ${TASK4} && make && make clean

task6 : ${TASK6}*.tex/ Makefile ${TASK6}Makefile
	cd ${TASK6} && make && make clean

generate : ${ALL_OBJ}
	-make clean
	-mkdir build
	-make task2
	-make task3
	-make task4

clean:
	cd ${TASK2} && make clean
	cd ${TASK3} && make clean
	cd ${TASK4} && make clean
	cd ${TASK6} && make clean
