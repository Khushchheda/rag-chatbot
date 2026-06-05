from concurrent.futures import ThreadPoolExecutor
import time


def search():

    time.sleep(3)

    return "Mumbai population: 12.5 million"


def calculate():

    time.sleep(2)

    return str(25 * 12)


with ThreadPoolExecutor() as executor:

    search_future = executor.submit(search)

    calc_future = executor.submit(calculate)

    search_result = search_future.result()

    calc_result = calc_future.result()


print(search_result)

print(calc_result)
