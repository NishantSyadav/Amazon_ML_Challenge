from src.evaluate import entity_f05


def almost_equal(a, b, eps=1e-8):
    return abs(a - b) < eps


# Test 1: Perfect multi-match
score = entity_f05(
    {"S2-1", "S3-2"},
    {"S2-1", "S3-2"}
)

assert almost_equal(score, 1.0)


# Test 2: Correct singleton
score = entity_f05(
    set(),
    set()
)

assert almost_equal(score, 1.0)


# Test 3: False match on singleton
score = entity_f05(
    set(),
    {"S2-1"}
)

assert almost_equal(score, 0.0)


# Test 4: Miss every true match
score = entity_f05(
    {"S2-1"},
    set()
)

assert almost_equal(score, 0.0)


# Test 5: Amazon-style example
#
# Truth:
# S2-47, S3-812
#
# Prediction:
# S2-47, S2-193, S3-812
#
# Precision = 2/3
# Recall = 1
# F0.5 = 0.714285...

score = entity_f05(
    {"S2-47", "S3-812"},
    {"S2-47", "S2-193", "S3-812"}
)

print("Amazon example F0.5:", score)

assert almost_equal(
    score,
    0.7142857142857143
)


print("\nALL EVALUATOR TESTS PASSED ✅")