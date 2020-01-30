from briscola.card import Card

cdef dict values_points = {1: 11, 2: 0, 3: 10, 4: 0, 5: 0, 6: 0, 7: 0, 8: 2, 9: 3, 10: 4}

cpdef int select_winner(list table, briscola):
    cdef:
        first = table[0]
        second = table[1]
        int first_points = values_points[first.value]
        int second_points = values_points[second.value]
        int first_briscola = first.seed == briscola.seed
        int second_briscola = second.seed == briscola.seed
    if first_briscola:
        first_points += 100
    if second_briscola:
        second_points += 100
    if not first_briscola and not second_briscola and second.seed != first.seed:
        second_points -= 100
    return second_points > first_points