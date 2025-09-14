export default function Background({
  children,
}: {
  children?: React.ReactNode;
}) {
  return (
    <section
      className={`
      h-svh w-svw flex items-center justify-center
      [background-size:90px_90px,90px_90px,90px_90px]
      [background-position:0_0,0_0,45px_45px]
      lg:[background-size:120px_120px,120px_120px,120px_120px]
      lg:[background-position:0_0,0_0,60px_60px]
      `}
      style={{
        backgroundColor: "#111",
        backgroundImage: `
      repeating-linear-gradient(
      to bottom,
      transparent,
      transparent 30px,
      rgba(0,0,0,0.02) 30px,
      rgba(0,0,0,0.02) 60px
      ),
      url('data:image/svg+xml;utf8,<svg stroke="currentColor" fill="white" fill-opacity="0.02" stroke-width="0" viewBox="0 0 60 60" height="32" width="32" xmlns="http://www.w3.org/2000/svg"><path fill="none" d="M0 0h24v24H0z"/><path d="M12 1 3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 10.99h7c-.53 4.12-3.28 7.79-7 8.94V12H5V6.3l7-3.11v8.8z"/></svg>'),
      url('data:image/svg+xml;utf8,<svg stroke="currentColor" fill="white" fill-opacity="0.02" stroke-width="0" viewBox="0 0 60 60" height="32" width="32" xmlns="http://www.w3.org/2000/svg"><path fill="none" d="M0 0h24v24H0z"/><path d="M12 1 3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 10.99h7c-.53 4.12-3.28 7.79-7 8.94V12H5V6.3l7-3.11v8.8z"/></svg>')
      `,
        backgroundRepeat: "repeat, repeat, repeat",
      }}
    >
      {children}
    </section>
  );
}
