# How to contribute

We definitely welcome patches and contribution to fastText.py!

Here are some guidelines and information about how to do so.

## Sending patches

### Getting started

1. Check out the code:

        $ git clone https://github.com/salestock/fastText.py.git
        $ cd fastText.py
        $ pip install -r requirements.txt

1. Create a fork of the fastText.py repository.
1. Add your fork as a remote:

        $ git remote add fork git@github.com:$YOURGITHUBUSERNAME/fastText.py.git

1. Make changes, commit them.
1. Run the test suite:

        $ make install-dev
        $ make test

1. Push your changes to your fork:

        $ git push fork ...

1. Open a pull request.

## Filing Issues
When filing an issue, make sure to answer these five questions:

1. What version of Python are you using (`python --version`)?
2. What version of `fasttext` are you using (`pip list |  grep fasttext`)?
3. What operating system and processor architecture are you using?
4. What did you do?
5. What did you expect to see?
6. What did you see instead?

### Contributing code
Unless otherwise noted, the fastText.py source files are distributed under
the BSD-style license found in the LICENSE file.
