import operator
from typing import List

from sudoku.matrix import IntMatrix, Matrix


class Solver:
    """
    Sudoku problem solver.
    """

    def __init__(self, puzzle: Matrix):
        self.puzzle: Matrix = puzzle
        self.answer: Matrix = None

    def solve(self) -> bool:
        flatten_puzzle: List[int] = [num for row in self.puzzle.matrix for num in row]
        is_solved, flatten_answer = sudokusolve(flatten_puzzle)
        if is_solved:
            answer_2d: IntMatrix = [flatten_answer[9 * i : 9 * i + 9] for i in range(9)]
            self.answer = Matrix(answer_2d)
            return True
        else:
            return False


blockdic = {0: 0, 1: 3, 2: 6, 3: 27, 4: 30, 5: 33, 6: 54, 7: 57, 8: 60}
result = [2**x for x in range(9)]
resultdic = dict(zip(result, [i for i in range(1, 10)]))


# solve function
def sudokusolve(s0_r):
    """
    s0_p中存储着确定的解，其中解未确定的位置值等于0
    s1_p中存储这未确定的解的可能值,其中解确定的位置值等于0
    """
    s0_p = [r2p(x) for x in s0_r]
    print("=" * 50)
    print("The SUDOKU question is:")
    printseq1(s0_r)  # 打印待解数独
    print("=" * 50)
    print("Begin to solve......\n")
    s0_p, s1_p = trybest(s0_p)
    if jobnotfinished(s0_p):  # 在第一次尽力求解后，需要先判断一下是否已经得出结果
        workout, s0_p = solvebyusablenum(s0_p)
        if workout == True:
            print("Successfully get a result:")
            printseq1([p2r_0(x) for x in s0_p])
            print(
                "\n********************************\n"
                "*Winner winner, chicken dinner!*\n"
                "********************************"
            )
            return True, [p2r_0(x) for x in s0_p]
        else:
            print("No solution exist!")
            return False, []
    else:
        print("Successfully get a result:")
        printseq1([p2r_0(x) for x in s0_p])
        print(
            "\n********************************\n"
            "*Winner winner, chicken dinner!*\n"
            "********************************"
        )
        return True, [p2r_0(x) for x in s0_p]


def trybest(seq0):
    seq0_save = seq0.copy()
    seq1 = [0 for x in range(81)]
    while True:
        seq0_save = seq0.copy()
        seq0, seq1 = solveonce(seq0)
        over = operator.eq(seq0, seq0_save)  # 判断是否尽力
        if over == True:
            # printcombin_r(seq0, seq1)
            break
    return seq0, seq1


def solvebyusablenum(seq0):
    """用可能的数值求解数独的最终解"""
    seq0, seq1 = solveonce(seq0)  # 为了产生seq1
    seq0_store = [0 for x in range(81)]
    for x1 in range(81):
        if seq0[x1] != 0:
            continue
        for x2 in usablenum(seq1[x1]):
            seq0_store = seq0.copy()
            seq0[x1] = x2
            # print("\nChange cell(%d,%d)'s value to %d, then coclude:"%(x1//9, x1%9, p2r_0(x2)))
            seq0, seq1 = trybest(seq0)
            if rulebreak(seq0):
                seq0 = seq0_store.copy()
                continue
            if jobnotfinished(seq0):
                findresult, seq0_h = solvebyusablenum(seq0)
                # print("findresult =", findresult)
                if findresult == False:
                    # print("上层更换下一个数")
                    seq0 = seq0_store.copy()
                    continue  # 下层递归返回表示求解失败，则该级递归继续试下一个可能值
                else:
                    # print("下级递归发现最终解，向上回归")
                    return (
                        True,
                        seq0_h,
                    )  # 下级递归返回表示求解成功，则直接向上级递归同样返回成功
            else:
                return True, seq0  # 最下级递归得出正确的最终解
        # print("下层求解失败")
        return (
            False,
            [],
        )  # [x1]位置所有可能数都不行，返回False指示上一级递归需要continue


def solveonce(seq0):
    seq1 = [0 for x in range(81)]
    seq1 = blocksolve(seq0, seq1)
    seq1 = rowcolumnsolve(seq0, seq1)
    for x in range(81):
        if seq1[x] in result:
            seq0[x], seq1[x] = seq1[x], 0
    return seq0, seq1


def blocksolve(seq0, seq1):
    for i0 in range(9):
        i = blockdic[i0]
        x = sum(block(seq0, i0)) ^ 511
        for j in [i, i + 1, i + 2, i + 9, i + 10, i + 11, i + 18, i + 19, i + 20]:
            if seq0[j] == 0:
                seq1[j] = x
    return seq1


def block(seq, num):
    i = blockdic[num]
    return seq[i : i + 3] + seq[i + 9 : i + 12] + seq[i + 18 : i + 21]


def rowcolumnsolve(seq0, seq1):
    for x in range(9):
        for y in range(9):
            if seq0[x * 9 + y] == 0:
                x1 = sum(row(seq0, x)) ^ 511
                y1 = sum(column(seq0, y)) ^ 511
                seq1[x * 9 + y] = seq1[x * 9 + y] & x1 & y1
    return seq1


def row(seq, num):
    return seq[num * 9 : (num + 1) * 9]


def column(seq, num):
    return seq[num::9]


# process control function
def jobnotfinished(seq0):
    """判断数独已经得出最终解"""
    return 0 in seq0  # 数组seq0还存在0,表明还不是最终解


def rulebreak(seq0):
    """判断数组解是否违反规则，如违反则说明违反原因，返回True"""
    # 判断块是否违反规则
    for x in range(9):
        if existsamenum(block(seq0, x)):
            # print("Block %d breaks the rule." % x)
            return True
    # 判断行是否违反规则
    for x in range(9):
        if existsamenum(row(seq0, x)):
            # print("Row %d breaks the rule." % x)
            return True
    # 判断列是否违反规则
    for x in range(9):
        if existsamenum(column(seq0, x)):
            # print("Column %d breaks the rule." % x)
            return True
    # 全不违反则返回False
    return False


# assistant function
def r2p(num_r):
    """
    1-->000000001  2-->000000010  3-->000000100
    4-->000001000  5-->000010000  6-->000100000
    7-->001000000  8-->010000000  9-->100000000
    0-->000000000 代表1～9均可
    """
    if num_r == 0:
        return 0
    else:
        return 2 ** (num_r - 1)


def p2r_0(num_p):
    return resultdic[num_p]


def p2r(num_p):
    if num_p == 0:
        return 123456789
    else:
        p_str = ""
        for x in range(1, 10):
            if num_p & 1:
                p_str = p_str + str(x)
            num_p = num_p >> 1
        return int(p_str)


def usablenum(num_p):
    return [x for x in result if num_p & x == x]


def existsamenum(seq):
    """判断列表seq中存在相同的元素（0除外）"""
    newlist = [x for x in seq if x != 0]
    setlist = list(set(newlist))
    return len(newlist) > len(setlist)  # 大于代表存在相同元素，相等代表不存在相同元素


def printseq(seq):
    """按数独格式打印某一数组"""
    for i in range(9):
        for j in range(9):
            n = 9 * i + j
            print("%-10d" % seq[n], end="")
        print("")
    # print('')


def printseq1(seq):
    """按数独格式小间距打印某一数组"""
    for i in range(9):
        for j in range(9):
            n = 9 * i + j
            print("%-2d" % seq[n], end="")
        print("")
    # print('')


def printcombin(seq0, seq1):
    """将两个数组合并成新的数组，然后按数独格式打印"""
    printseq([seq0[x] + seq1[x] for x in range(81)])


def printcombin_r(seq0, seq1):
    printseq([p2r(seq0[x] + seq1[x]) for x in range(81)])


def main():
    pass


if __name__ == "__main__":
    main()
