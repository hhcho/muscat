from enum import Enum

# Round protocol definitions
class prot(Enum):
    # Training
    UNIFY_LOC = 1 # unify locations
    LOCAL_STATS = 2 # compute infection stats and exposure loads
    LOCAL_STATS_SECURE = 3 # LOCAL_STATS with secure aggregation
    FEAT = 4 # construct training features (and feature stats for scaling)
    FEAT_SECURE = 5 # FEAT with secure aggregation of feature stats
    ITER_FIRST = 6 # model learning first iter
    ITER_FIRST_SECURE = 7 # ITER_FIRST with secure aggregation of gradients
    ITER = 8 # model learning iter
    ITER_SECURE = 9 # ITER with secure aggregation of gradients
    ITER_LAST = 10 # model learning last iter
    ITER_LAST_SECURE = 11 # ITER_LAST with secure aggregation of gradients
    
    # Prediction (test)
    PRED_LOCAL_STATS = 101 # compute infection stats and exposure loads
    PRED_LOCAL_STATS_SECURE = 102 # PRED_LOCAL_STATS with secure aggregation
    PRED_FEAT = 103 # construct test features
    PRED_FEAT_SECURE = 104 # PRED_FEAT with secure aggregation
    
    # Tests for debugging
    TEST_CPS = 201
    TEST_AGG_1 = 202
    TEST_AGG_2 = 203
    TEST_AGG_3 = 204
    

# Workflow definitions
class workflow:

    # Empty workflow
    EMPTY = []

    # Secure training workflow
    def SECURE_TRAIN_ROUNDS(num_iters: int):
        return [None,
                prot.UNIFY_LOC,
                prot.LOCAL_STATS_SECURE,
                prot.FEAT_SECURE,
                prot.ITER_FIRST_SECURE,
                *[prot.ITER_SECURE] * (num_iters - 2),
                prot.ITER_LAST_SECURE]

    # Plaintext training workflow
    def PLAIN_TRAIN_ROUNDS(num_iters: int):
        return [None,
                prot.UNIFY_LOC,
                prot.LOCAL_STATS,
                prot.FEAT,
                prot.ITER_FIRST,
                *[prot.ITER] * (num_iters - 2),
                prot.ITER_LAST]

    # Secure test workflow
    SECURE_TEST_ROUNDS = [None,
        prot.LOCAL_STATS_SECURE,
        prot.FEAT_SECURE
    ]

    PLAIN_TEST_ROUNDS = [None,
        prot.LOCAL_STATS,
        prot.FEAT
    ]

    # Debugging test workflow
    DEBUG_ROUNDS = [None,
        prot.TEST_AGG_1,
        prot.TEST_AGG_2,
        prot.TEST_AGG_3
    ]
