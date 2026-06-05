from langchain.tools import tool


@tool
def calculator(expression: str) -> str:
    """Useful for mathematical calculations."""

    try:
        return str(eval(expression))

    except Exception as e:
        return f"Calculation Error: {e}"


@tool
def search(query: str) -> str:
    """Useful for searching information."""

    if query.strip() == "":

        return "No search query provided."

    return f"Search result for: {query}"

print(
    calculator.invoke("25*12")
)

print(
    calculator.invoke("10/0")
)

print(
    search.invoke("")
)