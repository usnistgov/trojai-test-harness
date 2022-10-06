from setuptools import setup

setup(
    name='trojai-test-harness',
    version='1.0',
    packages=['data_science', 'data_science.one_off', 'leaderboards', 'local_evaluator'],
    url='https://github.com/usnistgov/trojai-test-harness',
    license='',
    author='Tim Blattner',
    author_email='timothy.blattner@nist.gov',
    install_requires=[
    'wheel',
    'google-api-python-client',
    'google-auth-httplib2',
    'google-auth-oauthlib',
    'jsonpickle',
    'jsonschema',
    'spython',
    'hypothesis-jsonschema',
    'pid',
    'numpy',
    'pytablewriter',
    'dominate',
    'GitPython',
    'httplib2',
    'sklearn',
    'airium',
    'pandas'
    ],
    description='Trojai Test Harness'
)
