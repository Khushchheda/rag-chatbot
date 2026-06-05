from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """Useful for mathematical calculations."""
    return str(eval(expression))

@tool
def search(query: str) -> str:
    """Useful for searching information."""
    return f"Search result for: {query}"


query = input("Ask a question: ")

# REASON

if any(char.isdigit() for char in query):

    print("\nThought:")
    print("This looks like a calculation.")

    print("\nAction:")
    print("Calculator")

    result = calculator.invoke(
        query.replace("What is", "")
    )

    print("\nObservation:")
    print(result)

    print("\nFinal Answer:")
    print(result)

else:

    print("\nThought:")
    print("This requires searching.")

    print("\nAction:")
    print("Search")

    result = search.invoke(query)

    print("\nObservation:")
    print(result)

    print("\nFinal Answer:")
    print(result)