import "./DotLoader.css";

const DotLoader = (props: { showAnimation: boolean }) => {
  return (
    <div
      className="dot-loader"
      style={{
        animation: props.showAnimation ? "blink 1s infinite" : "initial",
      }}
    >
      {" "}
      ●
    </div>
  );
};

export default DotLoader;
