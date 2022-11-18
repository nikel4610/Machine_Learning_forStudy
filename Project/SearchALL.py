from SearchKeywords import fm_search1, fm_search2, fm_search3
from multiprocessing import Process

if __name__ == "__main__":
    th1 = Process(target=fm_search1)
    th2 = Process(target=fm_search2)
    th3 = Process(target=fm_search3)

    th1.start()
    th2.start()
    th3.start()
    th1.join()
    th2.join()
    th3.join()
