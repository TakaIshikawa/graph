# Third-Party Notices

This project incorporates components from third-party open source projects. The original copyright notices and licenses for these components are provided below.

## Python Dependencies

This project depends on the following third-party Python packages. Each package is governed by its respective license terms:

### Core Dependencies

- **Pydantic** (MIT License)
  - Copyright: Pydantic Contributors
  - URL: https://github.com/pydantic/pydantic
  - License: https://github.com/pydantic/pydantic/blob/main/LICENSE

- **NetworkX** (BSD-3-Clause License)
  - Copyright: NetworkX Developers
  - URL: https://github.com/networkx/networkx
  - License: https://github.com/networkx/networkx/blob/main/LICENSE.txt

- **NumPy** (BSD-3-Clause License)
  - Copyright: NumPy Developers
  - URL: https://github.com/numpy/numpy
  - License: https://github.com/numpy/numpy/blob/main/LICENSE.txt

- **SciPy** (BSD-3-Clause License)
  - Copyright: SciPy Developers
  - URL: https://github.com/scipy/scipy
  - License: https://github.com/scipy/scipy/blob/main/LICENSE.txt

- **Typer** (MIT License)
  - Copyright: Sebastián Ramírez
  - URL: https://github.com/tiangolo/typer
  - License: https://github.com/tiangolo/typer/blob/master/LICENSE

- **MCP** (MIT License)
  - Copyright: Anthropic
  - URL: https://github.com/modelcontextprotocol/python-sdk
  - License: https://github.com/modelcontextprotocol/python-sdk/blob/main/LICENSE

- **PyYAML** (MIT License)
  - Copyright: Ingy döt Net, Kirill Simonov
  - URL: https://github.com/yaml/pyyaml
  - License: https://github.com/yaml/pyyaml/blob/master/LICENSE

- **python-dotenv** (BSD-3-Clause License)
  - Copyright: Saurabh Kumar
  - URL: https://github.com/theskumar/python-dotenv
  - License: https://github.com/theskumar/python-dotenv/blob/main/LICENSE

### Optional Dependencies

- **Voyage AI Python Client** (Apache-2.0 License)
  - Copyright: Voyage AI, Inc.
  - URL: https://github.com/voyage-ai/voyageai-python
  - License: https://github.com/voyage-ai/voyageai-python/blob/main/LICENSE
  - Note: Only required when using Voyage AI embeddings

- **OpenAI Python Client** (Apache-2.0 License)
  - Copyright: OpenAI
  - URL: https://github.com/openai/openai-python
  - License: https://github.com/openai/openai-python/blob/main/LICENSE
  - Note: Only required when using OpenAI embeddings

## License Summary

All third-party dependencies use permissive open source licenses (MIT, BSD-3-Clause, Apache-2.0) that are compatible with this project's Apache-2.0 license.

## Obtaining License Texts

Full license texts for each dependency can be obtained from:

1. The URLs listed above
2. The installed package metadata: `pip show <package-name>`
3. The package installation directory: `python -c "import <package>; print(<package>.__file__)"`

## Attribution Requirements

While the licenses of the dependencies listed above generally do not require attribution in binary distributions, we acknowledge and thank all contributors to these projects for their work.

## Notices

This project's use of third-party packages is subject to each package's respective license terms. Users of this project are responsible for ensuring compliance with all applicable license requirements when redistributing or modifying this software.

For questions about third-party licenses, please review each dependency's license file directly or contact the project maintainers.

---

**Last Updated**: 2026-04-24
