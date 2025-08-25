"""Console script for splicevo."""

import typer
from rich.console import Console

from splicevo import utils

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for splicevo."""
    console.print("Replace this message by putting your code into "
               "splicevo.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
