from langchain.tools import tool
from langchain_experimental.tools import PythonREPLTool

# ------------------
# Calculator Tool
# ------------------

@tool
def calculator(expression: str) -> str:
    """Useful for mathematical calculations."""
    return str(eval(expression))

# ------------------
# Search Tool
# ------------------

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Search result for: {query}"

# ------------------
# Python Tool
# ------------------

python_tool = PythonREPLTool()

# ------------------
# Test Tools
# ------------------

print("\nCalculator:")
print(calculator.invoke("25*12"))

print("\nSearch:")
print(search.invoke("Mumbai population"))

print("\nPython:")
print(
    python_tool.invoke(
        "print(10**2)"
    )
)