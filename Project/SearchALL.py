from SearchFM import fm_search1, fm_search2, fm_search3, fm_search4, fm_search5, fm_search6
from multiprocessing import Process

if __name__ == "__main__":
    th1 = Process(target=fm_search1)
    th2 = Process(target=fm_search2)
    th3 = Process(target=fm_search3)
    th4 = Process(target=fm_search4)
    th5 = Process(target=fm_search5)
    th6 = Process(target=fm_search6)

    th1.start()
    th2.start()
    th3.start()
    th4.start()
    th5.start()
    th6.start()
    th1.join()
    th2.join()
    th3.join()
    th4.join()
    th5.join()
    th6.join()
