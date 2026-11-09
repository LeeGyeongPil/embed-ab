# 보안 정책 / Security policy

보안 취약점을 발견하셨다면 **공개 이슈 대신** 이 레포의 **Security → Report a vulnerability**(비공개 취약점 신고)로 알려 주세요. 확인 후 답변드리고, 고친 버전을 릴리스하면서 함께 안내합니다.

If you find a security issue, please **do not open a public issue**. Use **Security → Report a vulnerability** in this repository to report it privately. We will reply after checking and announce the fix with a release.

embed-ab 는 서버를 열지 않는 명령줄 도구입니다. 외부로 나가는 요청은 사용자가 지정한 `ollama:`·`openai:` 엔드포인트와 `st:` 모델 다운로드뿐이고, API 키는 `OPENAI_API_KEY` 환경 변수로만 읽습니다. 평가 세트 문서가 지정한 엔드포인트로 전송된다는 점에 유의하세요.

embed-ab is a command-line tool and does not open a server. The only outbound requests go to the `ollama:` / `openai:` endpoints you choose (and `st:` model downloads); the API key is read only from the `OPENAI_API_KEY` environment variable. Note that your evaluation-set text is sent to the endpoint you choose.
