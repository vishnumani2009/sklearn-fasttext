NEW_VERSION=26e524e
CURRENT_VERSION=$(cat fasttext/cpp/LAST_COMMIT)

if [ "$NEW_VERSION" = "$CURRENT_VERSION" ]; then
    echo "fastText: The source code is uptodate!"
else
    echo "fastText: Updating ... "
    mv fasttext/cpp facebookresearch-fastText-${CURRENT_VERSION}

    if [ -d "facebookresearch-fastText-${NEW_VERSION}" ]; then
        cp -r facebookresearch-fasttext-${NEW_VERSION} fasttext/cpp
        echo $NEW_VERSION >> fasttext/cpp/LAST_COMMIT
    else
        if [ ! -d "${NEW_VERSION}.tar.gz" ]; then
            echo "fastText: Downloading the new version ... "
            wget https://api.github.com/repos/facebookresearch/fasttext/tarball/${NEW_VERSION}\
                -O ${NEW_VERSION}.tar.gz
            tar xzfv ${NEW_VERSION}.tar.gz
            cp -r facebookresearch-fasttext-${NEW_VERSION} fasttext/cpp
            echo $NEW_VERSION >> fasttext/cpp/LAST_COMMIT
        else
            echo "fastText: ${NEW_VERSION}.tar.gz exists"
            tar xzfv ${NEW_VERSION}.tar.gz
            cp -r facebookresearch-fasttext-${NEW_VERSION} fasttext/cpp
            echo $NEW_VERSION >> fasttext/cpp/LAST_COMMIT
        fi

        echo "fastText: ${CURRENT_VERSION} updated to ${NEW_VERSION}"
    fi
fi
