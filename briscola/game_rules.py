values_points = {1: 11, 2: 0, 3: 10, 4: 0, 5: 0, 6: 0, 7: 0, 8: 2, 9: 3, 10: 4}


def select_winner(table: list, briscola):
    first = table[0]
    second = table[1]
    first_points = values_points[first.value]
    second_points = values_points[second.value]
    first_briscola = first.seed == briscola.seed
    second_briscola = second.seed == briscola.seed
    if first_briscola:
        first_points += 100
    if second_briscola:
        second_points += 100
    if not first_briscola and not second_briscola and second.seed != first.seed:
        second_points -= 100
    return second_points > first_points
