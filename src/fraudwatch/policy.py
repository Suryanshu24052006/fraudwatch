# above BLOCK the payment is stopped, between review and block it goes to an analyst, below review it goes through.
# the review threshold is learned: the cost-minimising threshold on the validation set (models/thresholds.json).
# the block threshold is a business choice: on validation, scores >= 0.8 were fraud 35 times out of 36,
# so a genuine customer almost never gets blocked outright.
BLOCK_THRESHOLD = 0.8


def decide(p, review_threshold, block_threshold=BLOCK_THRESHOLD):
    if p >= block_threshold:
        return "block"
    if p >= review_threshold:
        return "review"
    return "allow"
