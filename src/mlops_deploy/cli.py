"""Console script for mlops_deploy."""
import mlops_deploy

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for mlops_deploy."""
    console.print("Replace this message by putting your code into "
               "mlops_deploy.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
