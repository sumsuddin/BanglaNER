import click


@click.command()
@click.argument("report_dir", type=click.Path(exists=True))
def main(report_dir):
    """Visualize the results of the model testing.

    :param report_dir: Path to the directory containing the report files.
    """
    print("done")

if __name__ == '__main__':
    main()