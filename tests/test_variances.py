from power.variances import (IntraSubjectParallel,
                             IntraSubjectCrossover,
                             IntraSubjectCV,
                             InterSubjectParallel,
                             InterSubjectCrossover,
                             TotalParallelNoReplication,
                             TotalParallelReplication,
                             Total2By2Crossover,
                             Total2By2MCrossover)


def test_intra_subject_parallel():
    """ See Chow et al. 9.1.1 """
    h = IntraSubjectParallel(n=None, m=3, stdev_wt=0.30, stdev_wr=0.45,
                             similarity_limit=1.1, alpha=0.05, power=0.8,
                             hypothesis='superiority')
    h.calculate()
    assert h.n == 13
    assert h.power > 0.8

    h = IntraSubjectParallel(n=13, m=3, stdev_wt=0.30, stdev_wr=0.45,
                             similarity_limit=1.1, alpha=0.05, power=None,
                             hypothesis='superiority')
    h.calculate()
    assert h.power > 0.8

    h = IntraSubjectParallel(n=13, m=3, stdev_wt=0.30, stdev_wr=0.45,
                             similarity_limit=1.1, alpha=None, power=0.8,
                             hypothesis='superiority')
    h.calculate()
    assert h.alpha < 0.05


def test_intra_subject_crossover():
    """ See Chow et al. 9.2.1 """
    h = IntraSubjectCrossover(n=None, m=2, stdev_wt=0.30, stdev_wr=0.45,
                              similarity_limit=1.1, alpha=0.05, power=0.8,
                              hypothesis='superiority')
    h.calculate()
    assert h.n == 14
    assert h.power > 0.8

    h = IntraSubjectCrossover(n=14, m=2, stdev_wt=0.30, stdev_wr=0.45,
                              similarity_limit=1.1, alpha=0.05, power=None,
                              hypothesis='superiority')
    h.calculate()
    assert h.power > 0.8

    h = IntraSubjectCrossover(n=14, m=2, stdev_wt=0.30, stdev_wr=0.45,
                              similarity_limit=1.1, alpha=None, power=0.8,
                              hypothesis='superiority')
    h.calculate()
    assert h.alpha < 0.05


def test_intra_subject_cv():
    """ See Chow et al. 9.2.1 """
    h = IntraSubjectCV(n=None, m=2, cv_t=0.70, cv_r=0.50,
                       margin=-0.1, alpha=0.05, power=0.8,
                       hypothesis='superiority', model='simple')
    h.calculate()
    assert h.n == 34
    assert h.power > 0.8

    h = IntraSubjectCV(n=34, m=2, cv_t=0.70, cv_r=0.50,
                       margin=-0.1, alpha=0.05, power=None,
                       hypothesis='superiority', model='simple')
    h.calculate()
    assert h.power > 0.8

    h = IntraSubjectCV(n=34, m=2, cv_t=0.70, cv_r=0.50,
                       margin=-0.1, alpha=None, power=0.8,
                       hypothesis='superiority', model='simple')
    h.calculate()
    assert h.alpha < 0.05

    """ See Chow et al. 9.2.2 """
    h = IntraSubjectCV(n=None, m=2, cv_t=0.70, cv_r=0.50,
                       stdev_r=0.35, stdev_t=0.3,
                       margin=-0.1, alpha=0.05, power=0.8,
                       hypothesis='superiority', model='conditional')
    h.calculate()
    assert h.n == 15
    assert h.power > 0.8

    h = IntraSubjectCV(n=15, m=2, cv_t=0.70, cv_r=0.50,
                       stdev_r=0.35, stdev_t=0.3,
                       margin=-0.1, alpha=0.05, power=None,
                       hypothesis='superiority', model='conditional')
    h.calculate()
    assert h.power > 0.8

    h = IntraSubjectCV(n=15, m=2, cv_t=0.70, cv_r=0.50,
                       stdev_r=0.35, stdev_t=0.3,
                       margin=-0.1, alpha=None, power=0.8,
                       hypothesis='superiority', model='conditional')
    h.calculate()
    assert h.alpha < 0.05


def test_inter_subject_parallel():
    """ See Chow et al. 9.3.1

    Chow has 74.  We get 75, due to rounding.
    """
    h = InterSubjectParallel(n=None, m=3, stdev_bt=0.30,
                             stdev_wt=0.20, stdev_br=0.40,
                             stdev_wr=0.30, alpha=0.05,
                             hypothesis='superiority',
                             power=0.8, similarity_limit=1.10)
    h.calculate()
    assert h.n == 75
    assert h.power > 0.8

    h = InterSubjectParallel(n=75, m=3, stdev_bt=0.30,
                             stdev_wt=0.20, stdev_br=0.40,
                             stdev_wr=0.30, alpha=0.05,
                             hypothesis='superiority',
                             power=None, similarity_limit=1.10)
    h.calculate()
    assert h.power > 0.8

    h = InterSubjectParallel(n=75, m=3, stdev_bt=0.30,
                             stdev_wt=0.20, stdev_br=0.40,
                             stdev_wr=0.30, alpha=None,
                             hypothesis='superiority',
                             power=0.8, similarity_limit=1.10)
    h.calculate()
    assert h.alpha < 0.05


def test_inter_subject_crossover():
    """ See Chow et al. 9.3.2

    Chow has 66.  We get 67, due to rounding.
    """
    h = InterSubjectCrossover(n=None, m=2, stdev_bt=0.30,
                              stdev_wt=0.20, stdev_br=0.40,
                              stdev_wr=0.30, alpha=0.05,
                              hypothesis='superiority', rho=0.75,
                              power=0.8, similarity_limit=1.10)
    h.calculate()
    assert h.n == 67
    assert h.power > 0.8

    h = InterSubjectCrossover(n=67, m=2, stdev_bt=0.30,
                              stdev_wt=0.20, stdev_br=0.40,
                              stdev_wr=0.30, alpha=0.05,
                              hypothesis='superiority', rho=0.75,
                              power=None, similarity_limit=1.10)
    h.calculate()
    assert h.power > 0.8

    h = InterSubjectCrossover(n=67, m=2, stdev_bt=0.30,
                              stdev_wt=0.20, stdev_br=0.40,
                              stdev_wr=0.30, alpha=None,
                              hypothesis='superiority', rho=0.75,
                              power=0.8, similarity_limit=1.10)
    h.calculate()
    assert h.alpha < 0.05


def test_total_parallel_no_replication():
    """ See Chow et al. 9.4.1 """
    h = TotalParallelNoReplication(n=None, stdev_tt=0.55, stdev_tr=0.75,
                                   hypothesis='superiority', alpha=0.05,
                                   power=0.8, similarity_limit=1.10)
    h.calculate()
    assert h.n == 40
    assert h.power > 0.8

    h = TotalParallelNoReplication(n=40, stdev_tt=0.55, stdev_tr=0.75,
                                   hypothesis='superiority', alpha=0.05,
                                   power=None, similarity_limit=1.10)
    h.calculate()
    assert h.power > 0.8

    h = TotalParallelNoReplication(n=40, stdev_tt=0.55, stdev_tr=0.75,
                                   hypothesis='superiority', alpha=None,
                                   power=0.8, similarity_limit=1.10)
    h.calculate()
    assert h.alpha < 0.05


def test_total_parallel_replication():
    """ See Chow et al. 9.4.2 """
    h = TotalParallelReplication(n=None, m=3, stdev_bt=0.30,
                                 stdev_wt=0.20, stdev_br=0.40,
                                 stdev_wr=0.30, alpha=0.05,
                                 hypothesis='superiority',
                                 power=0.8, similarity_limit=1.10)
    h.calculate()
    assert h.n == 28
    assert h.power > 0.8

    h = TotalParallelReplication(n=28, m=3, stdev_bt=0.30,
                                 stdev_wt=0.20, stdev_br=0.40,
                                 stdev_wr=0.30, alpha=0.05,
                                 hypothesis='superiority',
                                 power=None, similarity_limit=1.10)
    h.calculate()
    assert h.power > 0.8

    h = TotalParallelReplication(n=28, m=3, stdev_bt=0.30,
                                 stdev_wt=0.20, stdev_br=0.40,
                                 stdev_wr=0.30, alpha=None,
                                 hypothesis='superiority',
                                 power=0.8, similarity_limit=1.10)
    h.calculate()
    assert h.alpha < 0.05


def test_total_2_by_2_crossover():
    """ See Chow et al. 9.4.3 """
    h = Total2By2Crossover(n=None, rho=1, stdev_bt=0.30,
                           stdev_wt=0.20, stdev_br=0.40,
                           stdev_wr=0.30, alpha=0.05,
                           hypothesis='superiority',
                           power=0.8, similarity_limit=1.10)
    h.calculate()
    assert h.n == 31
    assert h.power > 0.8

    h = Total2By2Crossover(n=31, rho=1, stdev_bt=0.30,
                           stdev_wt=0.20, stdev_br=0.40,
                           stdev_wr=0.30, alpha=0.05,
                           hypothesis='superiority',
                           power=None, similarity_limit=1.10)
    h.calculate()
    assert h.power > 0.8

    h = Total2By2Crossover(n=31, rho=1, stdev_bt=0.30,
                           stdev_wt=0.20, stdev_br=0.40,
                           stdev_wr=0.30, alpha=None,
                           hypothesis='superiority',
                           power=0.8, similarity_limit=1.10)
    h.calculate()
    assert h.alpha < 0.05


def test_total_2_by_m_crossover():
    """ See Chow et al. 9.4.3

    Chow has 0.106 for stdev star.  We get 0.115.  This is an error in Chow. I
    have done the calcultion several ways and only get 0.115.
    """
    h = Total2By2MCrossover(n=None, m=2, rho=0.75, stdev_bt=0.30,
                            stdev_wt=0.20, stdev_br=0.40,
                            stdev_wr=0.30, alpha=0.05,
                            hypothesis='superiority',
                            power=0.8, similarity_limit=1.10)
    h.calculate()
    assert h.n == 24
    assert h.power > 0.8

    h = Total2By2MCrossover(n=24, m=2, rho=0.75, stdev_bt=0.30,
                            stdev_wt=0.20, stdev_br=0.40,
                            stdev_wr=0.30, alpha=0.05,
                            hypothesis='superiority',
                            power=None, similarity_limit=1.10)
    h.calculate()
    assert h.power > 0.8

    h = Total2By2MCrossover(n=24, m=2, rho=0.75, stdev_bt=0.30,
                            stdev_wt=0.20, stdev_br=0.40,
                            stdev_wr=0.30, alpha=None,
                            hypothesis='superiority',
                            power=0.8, similarity_limit=1.10)
    h.calculate()
    assert h.alpha < 0.05
