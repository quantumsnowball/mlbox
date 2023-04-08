#
# Test
#
test:
	@pytest tests/ --pdb --verbose
test-parallel:
	@pytest tests/ --workers auto --verbose
#
#
# Typecheck
#
# generally, do these check before each major commit
typecheck:
	@mypy --strict mlbox
typecheck-everything:
	@mypy .

