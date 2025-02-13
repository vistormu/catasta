from catasta import Archway
from pydantic import BaseModel


def main() -> None:
    # this example requires pydantic, fastapi, and uvicorn
    archway = Archway(
        path="models/fnn.pt",
        compile_method="torchscript",
    )

    class Data(BaseModel):
        s0: float
        s1: float
        s2: float
        s3: float

    archway.serve(
        host="145.94.127.212",
        port=8080,
        pydantic_model=Data,
    )


if __name__ == '__main__':
    main()
