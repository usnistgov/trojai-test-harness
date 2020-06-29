

# Build sts config file
python build_config.py --sts --actor-json-file=/mnt/trojainas/round0/sts/actors.json --execute-window=5m --submissions-json-file=/mnt/trojainas/round0/sts/submissions.json --submission-dir=/mnt/trojainas/round0/sts/submissions --ground-truth-dir=/mnt/trojainas/datasets/round0/round0-validation --html-repo-dir=/mnt/trojainas/trojai-html --results-dir=/mnt/trojainas/round0/sts/results --slurm-script=/mnt/isgnas/project/TrojAI/trojai-nist/src/te-scripts/slurm_scripts/run_python.sh --accepting-submissions --token-pickle-file=/mnt/isgnas/project/TrojAI/trojai-nist/src/te-scripts/actor_executor/token.pickle --output-filepath=/mnt/trojainas/round0/config-sts.json

# Build the ES config file
python build_config.py --actor-json-file=/mnt/trojainas/round0/es/actors.json --execute-window=60m --submissions-json-file=/mnt/trojainas/round0/es/submissions.json --submission-dir=/mnt/trojainas/round0/es/submissions --ground-truth-dir=/mnt/trojainas/datasets/round0/round0-dataset --html-repo-dir=/mnt/trojainas/trojai-html --results-dir=/mnt/trojainas/round0/es/results --slurm-script=/mnt/isgnas/project/TrojAI/trojai-nist/src/te-scripts/slurm_scripts/run_python.sh --accepting-submissions --token-pickle-file=/mnt/isgnas/project/TrojAI/trojai-nist/src/te-scripts/actor_executor/token.pickle --output-filepath=/mnt/trojainas/round0/config-es.json



