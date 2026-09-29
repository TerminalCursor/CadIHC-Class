def test_file() -> None:
    """# Test File Interface"""
    from histclass.interfaces.file import File

    fname = "tests/outputs/FileTest.log"
    f = File(fname)
    assert f.filename == fname
    f.create()
    assert True, "Creation test"
    assert f.read() == "", "Creation wipes file"
    message = "These are some bytes from the test\n"
    f.append(message)
    assert True, "Appending test"
    assert f.read() == message, "Message not written"
