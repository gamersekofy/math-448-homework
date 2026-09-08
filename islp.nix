{
  pythonPackages,
  fetchFromGitHub,
}:
pythonPackages.buildPythonPackage rec {
  pname = "ISLP";
  version = "0.0.0.dev0";

  src = fetchFromGitHub {
    owner = "intro-stat-learning";
    repo = "ISLP";
    rev = "a1f4e43ca88d4c9a6f186930181d628b9270015d";
    hash = "sha256-DUc1pv2c54514y4mbuGuah2XiDgW9lJtM1OhpTlmK+Y=";
  };

  pyproject = true;

  build-system = with pythonPackages; [
    setuptools
    wheel
    setuptools-scm
    sphinx
  ];

  # Minimal set for Boston/data-analysis workflow
  dependencies = with pythonPackages; [
    numpy
    scipy
    pandas
    matplotlib
    scikit-learn
    statsmodels
    lxml
  ];

  env.SETUPTOOLS_SCM_PRETEND_VERSION = version;
  env.SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ISLP = version;

  pythonImportsCheck = ["ISLP" "ISLP.models"];
  doCheck = false;

  # allow optional extras (torch/lifelines/pygam) to be absent for now
  dontCheckRuntimeDeps = true;
}
