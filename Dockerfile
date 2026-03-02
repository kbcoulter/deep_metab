FROM dnhem/proj_deepmetab:v0.1

RUN python -m pip uninstall -y fairseq || true

# Instead disable build isolation so it uses the deps we installed above.
RUN python -m pip install --no-cache-dir --no-build-isolation \
    "fairseq @ git+https://github.com/facebookresearch/fairseq.git@98ebe4f1ada75d006717d84f9d603519d8ff5579"

CMD ["/bin/bash"]
