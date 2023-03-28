#
# Typecheck
#
# generally, do these check before each major commit
typecheck:
	@mypy --strict mlbox
typecheck-everything:
	@mypy --strict .

