import os, sys, glob
from pathlib import Path
import papermill as pm
import shutil as sh

from . import utils


def main(argc, argv):
    base_paths = [
        (utils.get_project_path() / '.model' / 'MoNA' / 'joint_sigmoid', 'joint_vae*'),
    ]
    notebook_paths = [
        utils.get_project_path() / 'notebook' / 'specvae-10-vae-strip-analysis.ipynb',
        utils.get_project_path() / 'notebook' / 'specvae-10-vae-plot-latent-pca.ipynb',
    ]
    
    for base_path, pattern in base_paths:
        paths = glob.glob(str(base_path / pattern))
        names = list(map(lambda path: Path(path).name, paths))

        for name in names:
            for notebook_path in notebook_paths:
                html_path = notebook_path.with_suffix('.html')
                params = dict(dataset='MoNA', model_name=name, model_dir=str(base_path / name))

                print("Prepare and execute notebook: '%s'" % notebook_path)
                print("Parameters: ", params)
                try:
                    pm.execute_notebook(notebook_path, notebook_path, parameters=params)
                except pm.PapermillException as pmpe:
                    print("Error executing notebook: {0}".format(pmpe))
                    continue
                except Exception as e:
                    print("Error executing notebook! {0}".format(e))
                    continue

                print("Export notebook to HTML...")
                os.system('jupyter nbconvert --to html ' + str(notebook_path))

                print("Copy HTML file to model directory.")
                sh.move(str(html_path), str(base_path / name / html_path.name))
                print("----------------------------------")
    
    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv))
