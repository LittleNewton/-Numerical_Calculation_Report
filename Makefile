default: generate

TASK2 = tex/2_非线性方程求根/
TASK3 = tex/3_线性方程组的直接解法/
TASK4 = tex/4_解线性方程组的迭代法/
ALL_OBJ = 

task2 : ${TASK2}*.tex/ Makefile ${TASK2}Makefile
	cd ${TASK2} && make && make clean

task3 : ${TASK3}*.tex/ Makefile ${TASK3}Makefile
	cd ${TASK3} && make && make clean

task4 : ${TASK4}*.tex/ Makefile ${TASK4}Makefile
	cd ${TASK4} && make && make clean

generate : ${TASK2}*.tex/ Makefile ${TASK2}Makefile
	-make clean
	-mkdir build
	-make task2
	-make task3
	-make task4

clean:
	cd ${TASK2} && make clean
	cd ${TASK3} && make clean
