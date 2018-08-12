def lcs(source, target):
    m = len(source)
    n = len(target)
    counter = [[0] * (n + 1) for x in range(m + 1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if source[i] == target[j]:
                c = counter[i][j] + 1
                counter[i + 1][j + 1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(source[i - c + 1:i + 1])
                elif c == longest:
                    lcs_set.add(source[i - c + 1:i + 1])

    return lcs_set