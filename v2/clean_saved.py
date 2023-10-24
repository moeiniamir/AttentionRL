from datetime import datetime, timedelta
from pathlib import Path
import typer

app = typer.Typer()

@app.command()
def delete_older_than(days: int):
    for file in Path('saved/archive').glob('*'):
        if file.is_file() and 'max' not in file.name and datetime.now() - datetime.fromtimestamp(file.stat().st_ctime) > timedelta(days=days):
            file.unlink()
            

@app.command()
def archive_saved(max_too: bool = False):
    for file in Path('saved').glob('*'):
        if file.is_file() and (max_too or 'max' not in file.name):
            file.rename(Path('saved/archive') / file.name)
            
if __name__ == '__main__':
    app()
