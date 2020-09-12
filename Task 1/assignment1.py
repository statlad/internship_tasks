# tests
# inputArray = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
# data = [1,2,3,-9,5,7,-99,2,8]
# neg_data = [-1,-2,-3]

def findMaxSubArray_brute(inputArray):
    """Greedy option with Quadratic Time Complexity."""

    totalSum = 0
    maxSubArray = []

    count_negative = sum(n < 0 for n in inputArray)

    if count_negative == 0:
        maxSubArray = inputArray

    elif count_negative == len(inputArray):
        maxSubArray = max(inputArray)

    else:
        for i in range(len(inputArray) + 1):
            for j in range(len(inputArray) + 1):
                currentMaxSum = sum(inputArray[i:j])
                if currentMaxSum > totalSum:
                    totalSum = currentMaxSum
                    maxSubArray = inputArray[i:j]

    return maxSubArray

findMaxSubArray_brute(inputArray)

def findMaxSubArray(inputArray):
    """O(n) Time complexity option."""
    start =  0
    end = len(inputArray)
    max_at_i = max_current = inputArray[start]
    max_left_at_i = max_current = start
    max_right_so_far = start + 1

    for i in range(start + 1, end):

        if max_at_i > 0:
            max_at_i += inputArray[i]
        else:
            max_at_i = inputArray[i]
            max_left_at_i = i
        if max_at_i > max_current:
            max_current = max_at_i
            max_left_so_far = max_left_at_i
            max_right_so_far = i + 1

    return inputArray[max_left_so_far: max_right_so_far]

findMaxSubArray(inputArray)
